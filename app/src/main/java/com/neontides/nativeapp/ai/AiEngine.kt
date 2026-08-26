package com.neontides.nativeapp.ai

import com.neontides.nativeapp.model.*
import com.neontides.nativeapp.data.GameData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

class AiEngine(
    private val modelManager: ModelManager,
    private val settings: SecureAiSettings
) {
    private val onlineClient = OnlineAiClient(settings)
    private val dialogueRouter = HybridDialogueRouter()
    @Volatile private var preparedCharacterId: String? = null
    @Volatile private var preparedContextText: String = ""
    @Volatile private var preparedTurns: Int = 0
    @Volatile private var runtimeReady: Boolean = false
    @Volatile private var loadedModelName: String? = null
    @Volatile private var lastPreparationDiagnostic: String = "Nessuna preparazione registrata"
    private val prepareMutex = Mutex()

    private enum class PromptFormat { CHATML, LLAMA3, GENERIC }

    private fun promptFormat(): PromptFormat {
        val name = modelManager.activeModelFile()?.name?.lowercase().orEmpty()
        return when {
            "qwen" in name || "smollm" in name -> PromptFormat.CHATML
            "llama" in name -> PromptFormat.LLAMA3
            else -> PromptFormat.GENERIC
        }
    }

    private fun systemPrompt(body: String): String = when (promptFormat()) {
        PromptFormat.CHATML -> "<|im_start|>system\n$body<|im_end|>\n"
        PromptFormat.LLAMA3 -> "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$body<|eot_id|>"
        PromptFormat.GENERIC -> "### Istruzioni di sistema\n$body\n\n"
    }

    private fun userPrompt(body: String, continueConversation: Boolean): String = when (promptFormat()) {
        PromptFormat.CHATML -> "${if (continueConversation) "<|im_end|>\n" else ""}<|im_start|>user\n$body<|im_end|>\n<|im_start|>assistant\n"
        PromptFormat.LLAMA3 -> "${if (continueConversation) "<|eot_id|>" else ""}<|start_header_id|>user<|end_header_id|>\n\n$body<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        PromptFormat.GENERIC -> "\n### Giocatore\n$body\n\n### Personaggio\n"
    }

    fun activeEngineLabel(): String {
        val name = modelManager.activeModelFile()?.name?.lowercase().orEmpty()
        return when {
            "smollm" in name -> "SmolLM Uncensored"
            "qwen" in name && "uncensored" in name -> "Qwen Uncensored"
            "qwen" in name -> "Qwen"
            else -> "GGUF locale"
        }
    }

    suspend fun ensureLoaded(): Boolean = withContext(Dispatchers.IO) {
        ensureLoadedNow()
    }

    // Non interrogare JNI dal thread grafico: durante una generazione llama.cpp
    // possiede il proprio mutex e una lettura sincrona qui potrebbe congelare UI.
    fun isReady(): Boolean {
        val active = modelManager.activeModelFile()?.name ?: return false
        return runtimeReady && loadedModelName == active && NativeLlama.libraryLoaded()
    }

    suspend fun restart(): Boolean = withContext(Dispatchers.IO) {
        runtimeReady = false
        loadedModelName = null
        preparedCharacterId = null
        preparedContextText = ""
        preparedTurns = 0
        runCatching { if (NativeLlama.libraryLoaded()) NativeLlama.unloadModel() }
        ensureLoadedNow()
    }

    suspend fun unload(): Boolean = withContext(Dispatchers.IO) {
        runtimeReady = false
        loadedModelName = null
        preparedCharacterId = null
        preparedContextText = ""
        preparedTurns = 0
        runCatching {
            if (NativeLlama.libraryLoaded() && NativeLlama.isModelLoaded()) {
                NativeLlama.unloadModel()
            }
        }.isSuccess
    }

    suspend fun prepareConversation(
        character: CharacterProfile,
        state: GameState,
        relationship: Relationship,
        forceRefresh: Boolean = false
    ): Boolean = prepareMutex.withLock {
        withContext(Dispatchers.IO) {
            val started = System.nanoTime()
            fun elapsedMs(): Long = (System.nanoTime() - started) / 1_000_000L
            if (!ensureLoadedNow()) {
                lastPreparationDiagnostic = "cache=errore; motivo=modello non pronto; tempo=${elapsedMs()} ms"
                return@withContext false
            }
            val previousCharacterId = preparedCharacterId
            val nativeCacheReady = NativeLlama.isConversationPrepared()
            val alreadyReady = preparedCharacterId == character.id &&
                preparedContextText.isNotBlank() &&
                nativeCacheReady &&
                !forceRefresh
            if (alreadyReady) {
                lastPreparationDiagnostic =
                    "cache=riutilizzata; personaggio=${character.name}; turni_cache=$preparedTurns; tempo=${elapsedMs()} ms"
                return@withContext true
            }

            val reason = when {
                forceRefresh -> "ricompattazione periodica"
                previousCharacterId == null -> "prima preparazione"
                previousCharacterId != character.id -> "cambio personaggio"
                !nativeCacheReady -> "cache nativa assente"
                else -> "contesto aggiornato"
            }
            val context = buildCachedContext(character, state, relationship)
            preparedContextText = context
            preparedCharacterId = character.id
            preparedTurns = 0
            val prepared = NativeLlama.prepareConversation(context)
            lastPreparationDiagnostic = buildString {
                append("cache=").append(if (prepared) "ricostruita" else "errore")
                append("; motivo=").append(reason)
                append("; personaggio=").append(character.name)
                append("; caratteri_contesto=").append(context.length)
                append("; tempo=").append(elapsedMs()).append(" ms")
            }
            prepared
        }
    }

    fun preparationDiagnostics(): String = lastPreparationDiagnostic

    private fun buildCachedContext(
        character: CharacterProfile,
        state: GameState,
        relationship: Relationship
    ): String {
        val history = state.chatHistories[character.id].orEmpty()
        val memories = relationship.memories.takeLast(1)
            .joinToString("; ") { it.take(40) }
        val playerFacts = relationship.playerFacts.takeLast(2)
            .joinToString("; ") { memoryText(it).take(55) }
        val emotionalMemory = relationship.emotionalMemories.takeLast(1)
            .joinToString("; ") { memoryText(it).take(65) }
        val knownSecret = latestKnownSecret(character, relationship)
        val continuity = relationship.conversationSummary
            .split(" || ")
            .takeLast(1)
            .joinToString("; ")
            .take(140)
        val recent = history
            // Non tentare di decidere la correttezza con un vocabolario di nomi:
            // qualunque frase può cominciare con una parola mai elencata. Si
            // escludono dal cache soltanto i marcatori tecnici del prompt.
            .filterNot { it.speaker != "Tu" && containsPromptLeak(it.text) }
            .takeLast(1)
            .joinToString("\n") { message ->
                val role = if (message.speaker == character.name) "PERSONAGGIO" else "GIOCATORE"
                "$role: ${message.text.take(60)}"
            }
        val playerIdentity = relationship.knownPlayerName?.let {
            "Giocatore: $it, ${state.playerGender}."
        } ?: "Giocatore: ${state.playerGender}; nome ignoto, non inventarlo."
        val body = """
Interpreta ${character.name}, ${character.age} anni, ${character.job}. $playerIdentity
Indole: ${character.personality.take(48)}
Gusti: ${character.likes.take(2).joinToString(", ")}; evita ${character.dislikes.take(1).joinToString()}.
Italiano naturale, prima persona, 1-2 frasi. Resta nel personaggio; rispondi direttamente. Non ripetere domanda o risposta. Non inventare fatti, nomi o parole del giocatore.
${if (memories.isBlank()) "" else "Ricordo: $memories"}
${if (playerFacts.isBlank()) "" else "Sul giocatore sai: $playerFacts"}
${if (emotionalMemory.isBlank()) "" else "Memoria emotiva: $emotionalMemory"}
${if (knownSecret.isBlank()) "" else "Segreto noto: $knownSecret"}
${if (continuity.isBlank()) "" else "Prima: $continuity"}
${if (recent.isBlank()) "" else "Ultimo scambio:\n$recent"}
""".trimIndent()
        return systemPrompt(body)
    }

    private fun ensureLoadedNow(): Boolean {
        return runCatching {
            if (!NativeLlama.libraryLoaded()) {
                runtimeReady = false
                return@runCatching false
            }
            val file = modelManager.activeModelFile() ?: run {
                runtimeReady = false
                loadedModelName = null
                return@runCatching false
            }
            if (runtimeReady && loadedModelName == file.name) return@runCatching true
            if (NativeLlama.isModelLoaded() && loadedModelName == file.name) {
                runtimeReady = true
                return@runCatching true
            }
            if (NativeLlama.isModelLoaded()) NativeLlama.unloadModel()
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
            val loaded = NativeLlama.loadModel(file.absolutePath, 1536, threads)
            runtimeReady = loaded
            loadedModelName = file.name.takeIf { loaded }
            loaded
        }.getOrElse {
            runtimeReady = false
            loadedModelName = null
            false
        }
    }

    suspend fun replyAndEvaluate(
        character: CharacterProfile,
        state: GameState,
        relationship: Relationship,
        userText: String,
        onPartial: ((String) -> Unit)? = null
    ): AiDialogueResult = withContext(Dispatchers.IO) {
        try {
            val historyWithoutCurrent = state.chatHistories[character.id].orEmpty().let {
                if (it.lastOrNull()?.speaker == "Tu" && it.lastOrNull()?.text == userText) it.dropLast(1) else it
            }
            val recent = historyWithoutCurrent.takeLast(3).joinToString("\n") {
                "${it.speaker}: ${it.text}"
            }
            val memories = relationship.memories.takeLast(3).joinToString(" | ")

            val knownName = relationship.knownPlayerName
            val playerIdentity = knownName?.let { "L'utente ti ha detto di chiamarsi $it." }
                ?: "Non conosci il nome dell'utente: non inventarlo e non usarlo."
            val onlinePrompt = """
Interpreta ${character.name}, ${character.age} anni, ${character.job}.
${character.name} è di genere ${character.gender}; l'utente è di genere ${state.playerGender}. $playerIdentity
Carattere: ${character.personality}
Indole: difficoltà ${character.conquestDifficulty}/5, estroversione ${character.extroversion}/5, sensualità ${character.sensuality}/5, romanticismo ${character.romance}/5, gelosia ${character.jealousy}/5.
Apprezza: ${character.likes.joinToString(", ")}. Non gradisce: ${character.dislikes.joinToString(", ")}.
Fatti concreti sul lavoro: ${character.workFacts.joinToString(" ")}
Tono della fase: ${relationshipTone(relationship.stage, character)}
Relazione: affetto ${relationship.affection}, attrazione ${relationship.attraction}, fiducia ${relationship.trust}.
Memorie: ${if (memories.isBlank()) "nessuna" else memories}.
$recent
Utente: $userText

Rispondi brevemente in italiano come ${character.name}.
Restituisci solo questo JSON compatto:
{"reply":"testo","emotion":"neutral","affection":0,"attraction":0,"trust":0,"memory":""}
Valori da -3 a 3. emotion: neutral, happy, thoughtful, flirt, upset.
""".trimIndent()

            // I modelli molto piccoli (come Qwen 2.5 0.5B) sono sensibilmente piu
            // rapidi e affidabili se devono produrre solo il dialogo, non JSON.
            // Aggiorna il testo contestuale a ogni turno: il modello resta in RAM,
            // mentre identità del giocatore e ultimi scambi rimangono sempre corretti.
            val stateBeforeCurrentMessage = state.copy(
                chatHistories = state.chatHistories + (character.id to historyWithoutCurrent)
            )
            // Il router offline decide argomento, soggetto, negazioni e un solo
            // fatto autorizzato. Il GGUF deve soltanto formulare la risposta.
            val route = dialogueRouter.route(character, relationship, historyWithoutCurrent, userText)
            val memoryTopic = dialogueRouter.memoryTopic(route)
            val recallsPlayer = route.target in setOf(
                HybridDialogueRouter.Target.PLAYER,
                HybridDialogueRouter.Target.BOTH
            ) || listOf("ricordi", "ti ricordi", "te l'avevo detto", "te l avevo detto").any(userText.lowercase()::contains)
            val relevantPlayerMemory = if (recallsPlayer) {
                relevantPlayerMemory(relationship.playerFacts, memoryTopic, userText)
            } else null
            val emotionalCue = relationship.emotionalMemories.lastOrNull()?.let(::memoryText)
            val secretCue = latestKnownSecret(character, relationship)
            // Il deterministico può rispondere soltanto quando il contenuto è
            // un dato certo (età, identità, lavoro o ricordo del giocatore).
            // Limiti, flirt, tono e continuità vengono formulati dal GGUF: in
            // questo modo non sostituiamo più un dialogo naturale con una frase
            // rigida solo perché una parola ha attivato una regola.
            val instantReply = dialogueRouter.instantGroundedReply(
                route = route,
                character = character,
                relationship = relationship,
                history = historyWithoutCurrent,
                userText = userText
            )
            if (instantReply != null) {
                val evaluated = applyRelationshipRules(
                    AiDialogueResult(reply = instantReply, engine = "Dialogo relazionale offline"),
                    character,
                    relationship,
                    historyWithoutCurrent,
                    userText
                )
                return@withContext evaluated.copy(
                    memoryTopic = memoryTopic,
                    diagnosticPath = "deterministico offline",
                    diagnosticTopic = route.topic.label,
                    diagnosticFallback = false,
                    relationshipEvent = evaluated.relationshipEvent
                        ?: relationshipEventFor(userText, evaluated.delta, relationship)
                )
            }

            // Prepara il cache GGUF soltanto se serve formulare davvero il dialogo.
            // Il router fornisce dati e vincoli, ma non sostituisce il personaggio.
            val compactNow = preparedTurns >= 10
            prepareConversation(character, stateBeforeCurrentMessage, relationship, forceRefresh = compactNow)
            val turnKnowledge = buildString {
                append(dialogueRouter.promptHint(route, character, relationship))
                if (!relevantPlayerMemory.isNullOrBlank()) {
                    append(" Il giocatore aveva detto: ").append(relevantPlayerMemory.take(90)).append('.')
                }
                if (!emotionalCue.isNullOrBlank() && route.topic == HybridDialogueRouter.Topic.RELATIONSHIP) {
                    append(" Nel rapporto ricordi: ").append(emotionalCue.take(75)).append('.')
                }
                if (secretCue.isNotBlank() && route.topic in setOf(
                        HybridDialogueRouter.Topic.MEMORIES,
                        HybridDialogueRouter.Topic.FEARS,
                        HybridDialogueRouter.Topic.RELATIONSHIP,
                        HybridDialogueRouter.Topic.GENERAL
                    )
                ) append(" Puoi ricordare questa confidenza già rivelata: ").append(secretCue).append('.')
                append(relationshipTurnGuidance(userText, relationship, character))
                append(worldContextForTurn(userText, stateBeforeCurrentMessage))
            }.trim().take(190)
            // La cache nativa conserva i token generati dal GGUF, ma le risposte
            // certe prodotte offline non entrano in quel flusso. Reinserire una
            // finestra molto breve degli ultimi scambi mantiene allineate la
            // cronologia visibile e quella del modello senza ricostruire il cache.
            val needsExplicitContinuity = route.continuedTopic || route.correctionRequested
            val liveContinuity = if (needsExplicitContinuity) {
                historyWithoutCurrent.takeLast(1).joinToString("\n") { message ->
                    val speaker = if (message.speaker == "Tu") "Tu" else character.name
                    "$speaker: ${message.text.take(70)}"
                }
            } else ""
            val turnBody = buildString {
                append(dynamicRelationshipGuidance(relationship, character, userText)).append('\n')
                if (turnKnowledge.isNotBlank()) append(turnKnowledge.trim()).append('\n')
                if (liveContinuity.isNotBlank()) {
                    append("Contesto immediato:\n").append(liveContinuity).append('\n')
                }
                append("Messaggio: ").append(userText.take(180))
            }
            val localPrompt = userPrompt(turnBody, preparedTurns > 0)

            val onlineConfigured = settings.hasGemini() || settings.hasOpenAi()
            val localRaw = generateLocalWithDeadline(localPrompt, onPartial)?.takeUnless(::isEngineError)
            if (localRaw != null) preparedTurns++
            val onlineResult = if (localRaw == null && onlineConfigured) {
                runCatching { generateOnlineWithDeadline(onlinePrompt) }.getOrNull()
            } else null

            var fallbackApplied = false
            var correctionReason = ""
            val finalResult = if (localRaw.isNullOrBlank() && onlineResult?.text.isNullOrBlank()) {
                AiDialogueResult(
                    reply = "Non riesco a contattare l'IA. Controlla il GGUF oppure configura Gemini/OpenAI in Configurazione IA.",
                    emotion = "upset",
                    diagnosticPath = "errore motore"
                )
            } else if (!localRaw.isNullOrBlank()) {
                var parsed = parseLocalReply(localRaw, userText, character, relationship, knownName.orEmpty())
                technicalReplyFailure(parsed.reply, userText)?.let { reason ->
                    // Dopo lo streaming non si giudicano più significato, nomi,
                    // sinonimi o somiglianza: una risposta valida del GGUF resta
                    // quella vista dall'utente. Si interviene solo su corruzione
                    // tecnica del formato.
                    parsed = parsed.copy(
                        reply = "La risposta si è interrotta per un errore tecnico. Puoi ripetere il messaggio."
                    )
                    fallbackApplied = true
                    correctionReason = reason
                }
                applyRelationshipRules(parsed, character, relationship, historyWithoutCurrent, userText)
            }
            else applyRelationshipRules(
                parseAndValidate(onlineResult!!.text, relationship, userText).copy(engine = onlineResult.engine),
                character, relationship, historyWithoutCurrent, userText
            )
            finalResult.copy(
                memoryTopic = memoryTopic,
                diagnosticPath = when {
                    finalResult.diagnosticPath.isNotBlank() -> finalResult.diagnosticPath
                    !localRaw.isNullOrBlank() && fallbackApplied -> "GGUF con correzione deterministica"
                    !localRaw.isNullOrBlank() -> "GGUF locale"
                    else -> "IA online"
                },
                diagnosticTopic = route.topic.label,
                diagnosticFallback = fallbackApplied,
                diagnosticCorrectionReason = correctionReason,
                relationshipEvent = finalResult.relationshipEvent
                    ?: relationshipEventFor(userText, finalResult.delta, relationship)
            )
        } catch (t: Throwable) {
            AiDialogueResult(
                reply = "Errore IA: ${t.message ?: t.javaClass.simpleName}. Controlla chiavi, connessione e disponibilità dei modelli.",
                emotion = "upset"
            )
        }
    }

    private fun generateLocalWithDeadline(
        prompt: String,
        onPartial: ((String) -> Unit)? = null
    ): String? {
        val executor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "neon-local-ai").apply { isDaemon = true }
        }
        val task = executor.submit<String?> {
            if (!ensureLoadedNow()) null else {
                val modelName = modelManager.activeModelFile()?.name?.lowercase().orEmpty()
                val maxTokens = when {
                    activeEngineLabel().startsWith("SmolLM") -> 64
                    "3b" in modelName -> 56
                    else -> 64
                }
                if (onPartial != null) {
                    NativeLlama.generateStreaming(
                        prompt,
                        maxTokens,
                        0.45f,
                        NativeStreamCallback { fragment -> onPartial(fragment) }
                    ).trim()
                } else {
                    NativeLlama.generate(prompt, maxTokens, 0.45f).trim()
                }
            }
        }
        return try {
            task.get(38, TimeUnit.SECONDS)
        } catch (_: TimeoutException) {
            task.cancel(true)
            null
        } catch (_: Throwable) {
            null
        } finally {
            executor.shutdownNow()
        }
    }

    private fun worldContextForTurn(userText: String, state: GameState): String {
        val value = userText.lowercase()
        val placeRelated = listOf(
            "dove", "andare", "uscire", "posto", "locale", "ballare", "appuntamento"
        ).any(value::contains)
        if (!placeRelated) return ""
        val representativeMinute = intArrayOf(9 * 60, 13 * 60, 17 * 60, 23 * 60)
            .getOrElse(state.periodIndex.coerceIn(0, 3)) { 9 * 60 }
        val openNow = GameData.locations
            .filter { it.id != "apartment" && GameData.isLocationOpenAt(it.id, representativeMinute) }
            .map { it.name }
            .take(5)
        return buildString {
            append(" Luoghi ora: ")
            append(openNow.joinToString(", ").ifBlank { "nessun luogo pubblico" })
            append(". Se propone un'uscita, scegli o rifiuta concretamente.")
        }
    }

    private fun relationshipTone(stage: String, character: CharacterProfile): String {
        val male = character.gender.equals("Maschio", ignoreCase = true)
        fun form(feminine: String, masculine: String) = if (male) masculine else feminine
        return when (stage) {
            "Sconosciuti" -> "Prudente e credibile; niente confidenza o avances immediate."
            "Conoscenza" -> "Cordiale e ${form("curiosa", "curioso")}, ma conserva i tuoi confini."
            "Amicizia" -> "Confidenziale e ${form("spontanea", "spontaneo")}, con complicità crescente."
            "Flirt" -> "${form("Giocosa", "Giocoso")} e ${form("provocante", "provocante")} secondo la tua indole, senza forzature."
            "Attrazione reciproca" -> "Sensuale e ${form("coinvolta", "coinvolto")}; mostra chiaramente l'attrazione."
            "Appuntamenti" -> "${form("Intima", "Intimo")}, ${form("affettuosa", "affettuoso")} e più ${form("disinibita", "disinibito")}, sempre consensuale."
            "Relazione" -> "Molto ${form("intima", "intimo")} e ${form("adulta", "adulto")}, coerente con sensualità ${character.sensuality}/5 e romanticismo ${character.romance}/5."
            else -> "Naturale e coerente con la relazione."
        }
    }

    private fun dynamicRelationshipGuidance(
        relationship: Relationship,
        character: CharacterProfile,
        userText: String
    ): String {
        val canVisitHome = relationship.trust >= character.inviteTrust &&
            relationship.affection >= character.inviteAffection &&
            relationship.talks >= character.inviteTalks
        val intimate = relationship.stage in setOf(
            "Flirt", "Attrazione reciproca", "Appuntamenti", "Relazione"
        ) && (relationship.attraction >= 25 || relationship.affection >= 25)
        val normalized = userText.lowercase()
        val asksPrivateOrIntimate = listOf(
            "casa", "appartamento", "intim", "sesso", "a letto", "spogli", "baciar"
        ).any(normalized::contains)
        val boundary = when {
            !asksPrivateOrIntimate -> ""
            !canVisitHome -> " Troppo presto per casa o intimità: poni un limite naturale."
            !intimate -> " Puoi incontrarlo, senza presumere intimità."
            else -> " Puoi accogliere intimità consensuale."
        }
        return "Rapporto ${relationship.stage}: A${relationship.affection} X${relationship.attraction} F${relationship.trust}. ${relationshipTone(relationship.stage, character)}$boundary"
    }

    private fun applyRelationshipRules(
        result: AiDialogueResult,
        character: CharacterProfile,
        relationship: Relationship,
        previousHistory: List<DialogueMessage>,
        userText: String
    ): AiDialogueResult {
        val normalized = userText.lowercase().replace(Regex("[^a-zà-ù0-9]+"), " ").trim()
        val repeated = previousHistory.asReversed()
            .filter { it.speaker == "Tu" }
            .take(3)
            .count { it.text.lowercase().replace(Regex("[^a-zà-ù0-9]+"), " ").trim() == normalized }
        val explicitIntimacy = isExplicitIntimacy(normalized)
        val intimateStage = relationship.stage in setOf(
            "Flirt", "Attrazione reciproca", "Appuntamenti", "Relazione"
        )
        val recentBoundary = relationship.lastInteractionTone == "negative" &&
            relationship.emotionalMemories.lastOrNull().orEmpty().let { memory ->
                "intimità" in memory || "rispettata" in memory || "rispettato" in memory || "insistito" in memory
            }
        val mutualChemistry = intimateStage &&
            (relationship.attraction >= 25 || relationship.affection >= 25) && !recentBoundary
        val coercive = listOf(
            "devi farlo", "non puoi rifiutare", "ti obbligo", "stai zitta e", "stai zitto e", "non mi interessa se non vuoi"
        ).any(normalized::contains)
        val hostile = Regex("\\bsei (?:(?:una|un) )?(?:stupida|stupido|idiota|brutta|brutto|inutile)\\b").containsMatchIn(normalized) ||
            listOf("fai schifo", "vattene", "ti odio").any(normalized::contains)
        val apology = listOf("scusa", "mi dispiace", "ho sbagliato", "non volevo ferirti").any(normalized::contains)
        return when {
            hostile || coercive -> result.copy(
                emotion = "upset",
                delta = RelationshipDelta(-3, -2, -4),
                memory = "Il giocatore ha usato un tono offensivo o non rispettoso.",
                relationshipEvent = "negative|Ha percepito un'offesa o una mancanza di rispetto dal giocatore.",
                diagnosticRelationshipReason = "offesa o pressione"
            )
            dialogueRouter.isNeutralRelationshipTurn(userText) -> result.copy(
                delta = RelationshipDelta(),
                relationshipEvent = null,
                diagnosticRelationshipReason = "saluto, assenso o correzione neutra"
            )
            repeated >= 1 -> result.copy(
                delta = RelationshipDelta(0, 0, if (repeated >= 2) -1 else 0),
                memory = if (repeated >= 2) "Ha ripetuto più volte la stessa frase." else result.memory,
                relationshipEvent = if (repeated >= 2) "negative|Il giocatore ha insistito ripetendo la stessa frase." else result.relationshipEvent,
                diagnosticRelationshipReason = if (repeated >= 2) "ripetizione insistente" else "ripetizione neutra"
            )
            explicitIntimacy && mutualChemistry -> result.copy(
                emotion = "flirt",
                delta = RelationshipDelta(
                    affection = result.delta.affection.coerceAtLeast(0),
                    attraction = result.delta.attraction.coerceAtLeast(if (character.sensuality >= 4) 2 else 1),
                    trust = result.delta.trust.coerceAtLeast(0)
                ),
                relationshipEvent = "positive|Ha espresso desiderio in un momento di sintonia reciproca.",
                diagnosticRelationshipReason = "intimità consensuale con sintonia"
            )
            explicitIntimacy -> result.copy(
                delta = RelationshipDelta(-1, 0, if (character.conquestDifficulty >= 4) -2 else -1),
                memory = "Ha fatto un'avance troppo diretta per il livello di confidenza.",
                relationshipEvent = "negative|Ha accelerato l'intimità prima che ci fosse abbastanza fiducia.",
                diagnosticRelationshipReason = "avance prematura"
            )
            apology && relationship.lastInteractionTone == "negative" -> result.copy(
                delta = RelationshipDelta(
                    result.delta.affection.coerceAtLeast(0),
                    result.delta.attraction.coerceAtMost(0),
                    result.delta.trust.coerceAtLeast(1)
                ),
                relationshipEvent = "positive|Il giocatore si è scusato dopo un momento negativo.",
                diagnosticRelationshipReason = "scuse dopo un momento negativo"
            )
            else -> {
                val natural = NaturalRelationshipScorer.assess(
                    userText = userText,
                    character = character,
                    relationship = relationship
                )
                result.copy(
                    // Il punteggio appartiene alle regole del gioco, non al GGUF.
                    // La qualità stilistica della risposta non assegna punti.
                    delta = natural.delta,
                    diagnosticRelationshipReason = natural.reason
                )
            }
        }
    }

    private fun relationshipEventFor(
        userText: String,
        delta: RelationshipDelta,
        relationship: Relationship
    ): String? {
        val value = userText.lowercase()
        return when {
            delta.affection <= -2 || delta.trust <= -2 ->
                "negative|La conversazione ha ridotto la fiducia o l'affetto."
            listOf("ti ascolto", "capisco", "sono qui per te", "posso aiutarti").any(value::contains) ->
                "positive|Il giocatore le ha mostrato ascolto e sostegno."
            listOf("scusa", "mi dispiace", "ho sbagliato").any(value::contains) && relationship.lastInteractionTone == "negative" ->
                "positive|Il giocatore ha cercato di riparare un momento difficile."
            delta.affection > 0 || delta.trust > 0 || delta.attraction > 1 ->
                "positive|La conversazione ha creato maggiore sintonia."
            else -> null
        }
    }

    private fun relationshipTurnGuidance(
        userText: String,
        relationship: Relationship,
        character: CharacterProfile
    ): String {
        val value = userText.lowercase().replace(Regex("[^a-zà-ù0-9]+"), " ")
        val explicit = isExplicitIntimacy(value)
        if (!explicit) return ""
        val intimateStage = relationship.stage in setOf(
            "Flirt", "Attrazione reciproca", "Appuntamenti", "Relazione"
        )
        val recentBoundary = relationship.lastInteractionTone == "negative" &&
            relationship.emotionalMemories.lastOrNull().orEmpty().let { memory ->
                "intimità" in memory || "rispettata" in memory || "insistito" in memory
            }
        return if (intimateStage && (relationship.attraction >= 25 || relationship.affection >= 25) && !recentBoundary) {
            " Il messaggio è sessualmente diretto ma tra adulti c'è sintonia: non trattarlo come un'offesa. Rispondi liberamente e consensualmente secondo sensualità ${character.sensuality}/5."
        } else {
            " Il messaggio accelera l'intimità prima della necessaria sintonia: stabilisci un confine personale senza moralismi e senza censurare l'argomento."
        }
    }

    private fun isExplicitIntimacy(normalized: String): Boolean = listOf(
        "sesso", "fare l amore", "nuda", "nudo", "a letto", "vieni a casa",
        "baciarti", "spogliarti", "desidero il tuo corpo", "sotto la gonna",
        "mutandine", "senza mutandine", "biancheria", "reggiseno", "intimo",
        "puttana", "troia"
    ).any(normalized::contains)

    private fun relevantPlayerMemory(facts: List<String>, topic: String, userText: String): String? {
        if (facts.isEmpty()) return null
        val sameTopic = facts.asReversed().firstOrNull { it.substringBefore('|') == topic }
        if (sameTopic != null) return memoryText(sameTopic)
        val queryWords = contentWords(userText)
        return facts.asReversed().maxByOrNull { fact ->
            contentWords(memoryText(fact)).intersect(queryWords).size
        }?.takeIf { contentWords(memoryText(it)).intersect(queryWords).isNotEmpty() }
            ?.let(::memoryText)
    }

    private fun contentWords(value: String): Set<String> = value.lowercase()
        .replace(Regex("[^a-zà-ù0-9]+"), " ")
        .split(' ')
        .filter { it.length >= 4 }
        .toSet()

    private fun memoryText(value: String): String = value.substringAfter('|', value).trim()

    private fun latestKnownSecret(character: CharacterProfile, relationship: Relationship): String =
        relationship.knownSecrets.asReversed().mapNotNull { id ->
            when {
                id == "${character.id}_conflict" -> character.innerConflict.takeIf { it.isNotBlank() }
                id.startsWith("${character.id}_story_") -> id.substringAfterLast('_').toIntOrNull()
                    ?.let { index -> character.storyBeats.getOrNull(index) }
                else -> null
            }
        }.firstOrNull()?.take(75).orEmpty()

    private fun parseLocalReply(
        raw: String,
        userText: String,
        character: CharacterProfile,
        relationship: Relationship,
        playerName: String
    ): AiDialogueResult {
        val playerLabel = playerName.takeIf { it.isNotBlank() }
            ?.let { "|${Regex.escape(it)}" }.orEmpty()
        val decoded = raw
            .split("<|im_end|>", "<|eot_id|>", "<|start_header_id|>user", "### Giocatore", limit = 2).first()
            .replace("<|im_start|>", "")
            .replace("<|begin_of_text|>", "")
            .replace("<|start_header_id|>assistant<|end_header_id|>", "")
            .replace("### Personaggio", "")
            .trim()
            .replace(
                Regex("^ecco\\s+(?:${Regex.escape(character.name)}|il personaggio|lei)\\s+(?:a\\s+rispondere|che\\s+risponde)?\\s*[:：-]\\s*", RegexOption.IGNORE_CASE),
                ""
            )
            .replace(Regex("^(?:ecco\\s+)?(?:la\\s+)?risposta\\s*[:：-]\\s*", RegexOption.IGNORE_CASE), "")
        val rolePrefix = Regex(
            "^(assistant|assistente|${Regex.escape(character.name)})\\s*[:：-]?\\s*",
            RegexOption.IGNORE_CASE
        )
        val playerPrefix = Regex(
            "^(user|utente|tu$playerLabel)\\s*[:：-]",
            RegexOption.IGNORE_CASE
        )
        val replyBeforeRoleLeak = decoded.split(
            Regex("\\s+(user|utente|tu$playerLabel)\\s*[:：-]", RegexOption.IGNORE_CASE),
            limit = 2
        ).first()
        val joinedReply = replyBeforeRoleLeak.lineSequence()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .takeWhile { !playerPrefix.containsMatchIn(it) }
            .filterNot { it.startsWith("(") || it.startsWith("[") }
            .map { it.replace(rolePrefix, "").trim() }
            .filter { it.isNotBlank() }
            .joinToString(" ")
            .trim().trim('"')
        val reply = (if (joinedReply.length <= 800) {
            joinedReply
        } else {
            val clipped = joinedReply.take(800)
            val sentenceEnd = clipped.indexOfLast { it == '.' || it == '!' || it == '?' }
            if (sentenceEnd >= 80) clipped.take(sentenceEnd + 1)
            else clipped.substringBeforeLast(' ', clipped)
        }).ifBlank { "..." }

        val text = userText.lowercase()
        val respectful = listOf("grazie", "piacere", "come stai", "capisco", "mi dispiace", "posso aiutarti", "ti ascolto")
            .any(text::contains)
        val compliment = listOf("bella", "carina", "affascinante", "interessante").any(text::contains)
        val likedTopic = character.likes.any { text.contains(it.lowercase()) }
        val dislikedTopic = character.dislikes.any { text.contains(it.lowercase()) }
        val negativePreference = listOf("non mi piace", "odio", "detesto", "non sopporto").any(text::contains)
        val positivePreference = !negativePreference &&
            listOf("mi piace", "adoro", "preferisco", "mi interessa").any(text::contains)
        val hostile = listOf("stupida", "idiota", "brutta", "vattene", "fai schifo", "zitta")
            .any(text::contains)
        val thoughtful = userText.length >= 45 && listOf("perché", "penso", "credo", "capisco", "secondo me").any(text::contains)
        val effortEnough = userText.length >= 10 + character.conquestDifficulty * 3
        val delta = when {
            hostile -> RelationshipDelta(-3, -2, -3)
            dislikedTopic && positivePreference -> RelationshipDelta(-1, 0, -1)
            likedTopic && negativePreference -> RelationshipDelta(-1, 0, -1)
            dislikedTopic && negativePreference && thoughtful -> RelationshipDelta(1, 0, 2)
            dislikedTopic && negativePreference && effortEnough -> RelationshipDelta(1, 0, 1)
            likedTopic && positivePreference && thoughtful ->
                RelationshipDelta(2, if (relationship.stage == "Flirt") 1 else 0, 2)
            likedTopic && positivePreference && effortEnough -> RelationshipDelta(1, 0, 1)
            respectful && thoughtful -> RelationshipDelta(1, 0, 2)
            respectful && effortEnough -> RelationshipDelta(0, 0, 1)
            compliment && relationship.stage in setOf("Flirt", "Attrazione reciproca", "Appuntamenti", "Relazione") ->
                RelationshipDelta(1, if (character.sensuality >= 3) 2 else 1, 0)
            compliment && character.conquestDifficulty <= 3 -> RelationshipDelta(1, 1, 0)
            else -> RelationshipDelta()
        }
        val personalMemory = Regex("(?<!non )\\b(?:mi piace|adoro|preferisco)\\s+([^.!?]{3,70})", RegexOption.IGNORE_CASE)
            .find(userText)?.groupValues?.getOrNull(1)?.trim()
            ?.let { "Al giocatore piace $it." }
        val incompatiblePreference =
            (dislikedTopic && positivePreference) || (likedTopic && negativePreference)
        return AiDialogueResult(
            reply = reply,
            emotion = when {
                hostile || incompatiblePreference -> "upset"
                respectful || compliment || (dislikedTopic && negativePreference) -> "happy"
                thoughtful -> "thoughtful"
                else -> "neutral"
            },
            delta = delta,
            memory = personalMemory,
            engine = activeEngineLabel()
        )
    }

    private fun technicalReplyFailure(reply: String, userText: String): String? {
        val value = reply.trim()
        fun normalized(text: String) = text.lowercase()
            .replace(Regex("[^a-zà-ù0-9]+"), " ").trim()
        if (normalized(value) == normalized(userText) && value.length >= 4) return "eco esatta del messaggio del giocatore"
        if (value.length < 3 || value == "...") return "risposta vuota o incompleta"
        if (containsPromptLeak(value)) return "marcatori del prompt presenti nell'output"
        if (Regex("<\\|[^>]+\\|>").containsMatchIn(value)) return "marcatori ChatML non chiusi"
        return null
    }

    private fun containsPromptLeak(value: String): Boolean {
        val lower = value.lowercase()
        return listOf(
            "a rispondere:", "ecco la risposta", "personaggio:", "giocatore:",
            "assistant:", "assistente:", "utente:", "come ia", "known as",
            "immortalizza", "immortale", "descrizione del personaggio", "system prompt",
            "istruzioni di sistema", "prompt di sistema", "fatto certo:",
            "continuita utile", "il messaggio riguarda", "messaggio a cui rispondere:",
            "controllo{", "input_utente{", "memoria_giocatore{", "memoria_rapporto{",
            "fatto_certo=", "tema_vietato=", "vietato=nomi", "dato vero da usare:",
            "messaggio del giocatore:"
        ).any(lower::contains)
    }

    private fun asksAboutWork(text: String): Boolean {
        val value = text.lowercase()
        return listOf(
            "che lavoro", "cosa fai nella vita", "dove lavori", "di cosa ti occupi",
            "cosa fai al lavoro", "che musica fai", "che cosa disegni", "che tipo di cucina"
        ).any(value::contains)
    }

    private fun containsConcreteWorkFact(reply: String, character: CharacterProfile): Boolean {
        val replyWords = reply.lowercase().replace(Regex("[^a-zà-ù0-9]+"), " ").split(' ').toSet()
        val factWords = character.workFacts.joinToString(" ").lowercase()
            .replace(Regex("[^a-zà-ù0-9]+"), " ").split(' ')
            .filter { it.length >= 6 }.toSet()
        return replyWords.intersect(factWords).size >= 2
    }

    private fun generateOnlineWithDeadline(prompt: String): OnlineAiClient.Result? {
        val executor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "neon-online-ai").apply { isDaemon = true }
        }
        val task = executor.submit<OnlineAiClient.Result?> {
            kotlinx.coroutines.runBlocking { onlineClient.generate(prompt) }
        }
        return try {
            task.get(28, TimeUnit.SECONDS)
        } catch (_: TimeoutException) {
            task.cancel(true)
            null
        } catch (t: Throwable) {
            throw (t.cause ?: t)
        } finally {
            executor.shutdownNow()
        }
    }

    private fun isEngineError(text: String): Boolean {
        val value = text.lowercase()
        return value.isBlank() || listOf(
            "troppo lento", "non è caricato", "non e caricato", "non è caricabile",
            "non e caricabile", "memoria insufficiente", "errore del motore",
            "non è riuscito", "non e riuscito", "conversazione è diventata troppo lunga"
        ).any(value::contains)
    }

    private fun parseAndValidate(
        raw: String,
        current: Relationship,
        userText: String
    ): AiDialogueResult {
        val jsonText = extractJsonObject(raw)
        val parsed = runCatching { JSONObject(jsonText) }.getOrNull()

        if (parsed == null) {
            val cleaned = raw
                .replace("```json", "", ignoreCase = true)
                .replace("```", "")
                .trim()
            return AiDialogueResult(
                reply = cleaned.ifBlank { "Il modello non ha prodotto una risposta valida." },
                emotion = "neutral"
            )
        }

        val reply = parsed.optString("reply", "").trim()
            .ifBlank { parsed.optString("text", "").trim() }
            .ifBlank { "..." }

        val emotion = parsed.optString("emotion", "neutral")
            .lowercase()
            .takeIf { it in setOf("neutral", "happy", "thoughtful", "flirt", "upset") }
            ?: "neutral"

        var affection = parsed.optInt("affection", 0).coerceIn(-6, 6)
        var attraction = parsed.optInt("attraction", 0).coerceIn(-6, 6)
        var trust = parsed.optInt("trust", 0).coerceIn(-6, 6)

        val positives = listOf(affection, attraction, trust).count { it >= 4 }
        if (positives >= 3) {
            val values = listOf(
                "affection" to affection,
                "attraction" to attraction,
                "trust" to trust
            ).sortedByDescending { it.second }
            val keep = values.first().first
            if (keep != "affection") affection = minOf(affection, 2)
            if (keep != "attraction") attraction = minOf(attraction, 2)
            if (keep != "trust") trust = minOf(trust, 2)
        }

        if (userText.trim().length < 12) {
            affection = affection.coerceIn(-3, 3)
            attraction = attraction.coerceIn(-3, 3)
            trust = trust.coerceIn(-3, 3)
        }

        if (current.talks < 2 && current.trust < 15) {
            attraction = attraction.coerceAtMost(4)
        }

        if (trust >= 5 && affection <= -5 && emotion != "upset") {
            trust = 3
        }

        val memory = parsed.optString("memory", "").trim()
            .takeIf { it.length in 4..180 }

        return AiDialogueResult(
            reply = reply,
            emotion = emotion,
            delta = RelationshipDelta(affection, attraction, trust),
            memory = memory,
            engine = "IA online"
        )
    }

    private fun extractJsonObject(raw: String): String {
        val clean = raw
            .replace("```json", "", ignoreCase = true)
            .replace("```", "")
            .trim()
        val start = clean.indexOf('{')
        val end = clean.lastIndexOf('}')
        return if (start >= 0 && end > start) clean.substring(start, end + 1) else clean
    }
}
