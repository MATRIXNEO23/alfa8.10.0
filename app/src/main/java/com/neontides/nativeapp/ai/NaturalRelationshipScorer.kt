package com.neontides.nativeapp.ai

import com.neontides.nativeapp.model.CharacterProfile
import com.neontides.nativeapp.model.Relationship
import com.neontides.nativeapp.model.RelationshipDelta

internal data class RelationshipAssessment(
    val delta: RelationshipDelta,
    val reason: String
)

/** Valuta il significato sociale del turno senza affidare lo stato del gioco al GGUF. */
internal object NaturalRelationshipScorer {
    fun assess(
        userText: String,
        character: CharacterProfile,
        relationship: Relationship
    ): RelationshipAssessment {
        val value = userText.lowercase().replace(Regex("[^a-zà-ù0-9]+"), " ").trim()
        if (value.length < 3) return RelationshipAssessment(RelationshipDelta(), "messaggio troppo breve")

        val supportive = listOf(
            "ti ascolto", "capisco", "sono qui per te", "posso aiutarti", "mi dispiace",
            "rispetto la tua scelta", "prenditi il tuo tempo"
        ).any(value::contains)
        val selfDisclosure = listOf(
            "mi chiamo", "ho ", "mi piace", "non mi piace", "io sono", "lavoro come",
            "il mio mestiere", "la mia professione", "vivo a", "abito a",
            "penso che", "mi sento", "ho paura", "il mio sogno"
        ).any(value::contains)
        val compliment = listOf("bella", "bello", "carina", "carino", "affascinante")
            .any { adjective -> Regex("\\bsei(?: davvero| molto| proprio)? $adjective\\b").containsMatchIn(value) } ||
            listOf("mi piaci", "adoro il tuo", "adoro la tua").any(value::contains)
        val touchesLike = character.likes.any { fact ->
            fact.lowercase().replace(Regex("[^a-zà-ù0-9]+"), " ")
                .split(' ').filter { it.length >= 4 }.any(value::contains)
        }
        val touchesDislike = character.dislikes.any { fact ->
            fact.lowercase().replace(Regex("[^a-zà-ù0-9]+"), " ")
                .split(' ').filter { it.length >= 4 }.any(value::contains)
        }
        val negativePreference = listOf("non mi piace", "odio", "detesto", "non sopporto").any(value::contains)
        val positivePreference = !negativePreference &&
            listOf("piace", "adoro", "preferisco", "mi interessa").any(value::contains)
        val interestedQuestion = userText.trim().endsWith('?') || listOf(
            "come stai", "come ti senti", "cosa pensi", "che ne pensi", "cosa ti piace", "ti piace",
            "che musica", "tempo libero", "raccontami", "parlami di te", "come mai",
            "qual e il tuo sogno", "qual è il tuo sogno", "cosa desideri", "cosa ti preoccupa",
            "della tua famiglia", "cosa fai", "che lavoro", "dove vorresti", "cosa ascolti"
        ).any(value::contains)

        if ((touchesDislike && positivePreference) || (touchesLike && negativePreference)) {
            return RelationshipAssessment(
                RelationshipDelta(affection = -1, trust = -1),
                "preferenza incompatibile"
            )
        }
        val matureStage = relationship.stage !in setOf("Sconosciuti", "Conoscenza")
        return when {
            supportive -> RelationshipAssessment(RelationshipDelta(trust = 1), "ascolto o sostegno")
            touchesDislike && negativePreference -> RelationshipAssessment(RelationshipDelta(affection = 1, trust = 1), "antipatia condivisa")
            touchesLike && positivePreference -> RelationshipAssessment(RelationshipDelta(affection = 1, trust = 1), "interesse condiviso")
            compliment && matureStage -> RelationshipAssessment(
                RelationshipDelta(affection = 1, attraction = if (character.sensuality >= 3) 1 else 0),
                "complimento coerente con il rapporto"
            )
            compliment -> RelationshipAssessment(RelationshipDelta(affection = 1), "complimento rispettoso")
            selfDisclosure && userText.length >= 8 -> RelationshipAssessment(RelationshipDelta(trust = 1), "confidenza personale")
            interestedQuestion -> RelationshipAssessment(RelationshipDelta(trust = 1), "domanda interessata")
            else -> RelationshipAssessment(RelationshipDelta(), "interazione ordinaria neutra")
        }
    }
}
