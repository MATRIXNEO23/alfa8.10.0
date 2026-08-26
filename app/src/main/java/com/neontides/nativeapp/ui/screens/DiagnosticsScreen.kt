package com.neontides.nativeapp.ui.screens

import android.os.Build
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import com.neontides.nativeapp.ai.ModelManager
import com.neontides.nativeapp.ai.NativeLlama
import com.neontides.nativeapp.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun DiagnosticsScreen(
    manager: ModelManager,
    appSummary: String,
    onRestartAi: () -> Unit,
    onClearAppDiagnostics: () -> Unit,
    onBack: () -> Unit
) {
    val clipboard = LocalClipboardManager.current
    val scope = rememberCoroutineScope()
    var refresh by remember { mutableIntStateOf(0) }
    var report by remember { mutableStateOf("Raccolta diagnostica in corso…") }
    var loading by remember { mutableStateOf(true) }
    var copied by remember { mutableStateOf(false) }

    LaunchedEffect(refresh, appSummary) {
        loading = true
        copied = false
        report = withContext(Dispatchers.IO) {
            val active = manager.activeModelFile()
            val runtime = Runtime.getRuntime()
            val nativeLoaded = NativeLlama.libraryLoaded()
            val modelLoaded = if (nativeLoaded) runCatching { NativeLlama.isModelLoaded() }.getOrDefault(false) else false
            val nativeLog = if (nativeLoaded) runCatching { NativeLlama.getDiagnostics() }
                .getOrElse { "Errore lettura diagnostica: ${it.message}" }
            else "La libreria neontides_llm non è stata caricata."

            """
NEON TIDES - DIAGNOSTICA IA
Versione app: ${BuildConfig.VERSION_NAME}
Dispositivo: ${Build.MANUFACTURER} ${Build.MODEL}
Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})
Processori disponibili: ${runtime.availableProcessors()}
Memoria JVM usata: ${(runtime.totalMemory() - runtime.freeMemory()) / 1048576} MB
Memoria JVM massima: ${runtime.maxMemory() / 1048576} MB

Libreria llama.cpp: ${if (nativeLoaded) "CARICATA" else "NON CARICATA"}
Modello in RAM: ${if (modelLoaded) "SÌ" else "NO"}
Modello attivo: ${active?.name ?: "nessuno"}
Dimensione modello: ${active?.let { "${it.length() / 1048576} MB" } ?: "-"}

RIEPILOGO APP
$appSummary

RIEPILOGO NATIVO
${compactNativeSummary(nativeLog)}

REGISTRO NATIVO
$nativeLog
            """.trimIndent()
        }
        loading = false
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(Color(0xFF090B13))
            .verticalScroll(rememberScrollState())
            .padding(20.dp)
    ) {
        Text("DIAGNOSTICA IA", color = Color(0xFFFF5A9E), style = MaterialTheme.typography.headlineMedium)
        Text("Dopo il test premi AGGIORNA, poi COPIA RISULTATO e incollalo nella chat.", style = MaterialTheme.typography.bodySmall)
        Spacer(Modifier.height(12.dp))
        Button(
            onClick = {
                clipboard.setText(AnnotatedString(report))
                copied = true
            },
            enabled = !loading,
            modifier = Modifier.fillMaxWidth()
        ) { Text(if (copied) "RISULTATO COPIATO" else "COPIA RISULTATO") }
        Spacer(Modifier.height(8.dp))
        OutlinedButton(
            onClick = { refresh++ },
            enabled = !loading,
            modifier = Modifier.fillMaxWidth()
        ) { Text(if (loading) "RACCOLTA…" else "AGGIORNA RISULTATO") }
        Spacer(Modifier.height(8.dp))
        Text(
            "Confronto consigliato: prova «quanti anni hai?» e poi una domanda libera più complessa. La copia conserva gli ultimi sei passaggi.",
            style = MaterialTheme.typography.bodySmall,
            color = Color(0xFFB9B4CA)
        )
        Spacer(Modifier.height(16.dp))

        Surface(color = Color(0xFF171927), shape = MaterialTheme.shapes.large) {
            Text(report, modifier = Modifier.fillMaxWidth().padding(14.dp), style = MaterialTheme.typography.bodySmall)
        }

        Spacer(Modifier.height(14.dp))
        Button(
            onClick = {
                onRestartAi()
                report += "\n\nRiavvio IA richiesto. Attendi qualche secondo, poi premi AGGIORNA."
            },
            enabled = !loading && manager.activeModelFile() != null,
            modifier = Modifier.fillMaxWidth()
        ) { Text("RIAVVIA IA") }
        Spacer(Modifier.height(8.dp))
        TextButton(
            onClick = {
                scope.launch {
                    withContext(Dispatchers.IO) {
                        if (NativeLlama.libraryLoaded()) runCatching { NativeLlama.clearDiagnostics() }
                    }
                    onClearAppDiagnostics()
                    refresh++
                }
            },
            enabled = !loading,
            modifier = Modifier.fillMaxWidth()
        ) { Text("PULISCI REGISTRO") }
        Spacer(Modifier.height(8.dp))
        OutlinedButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("← MENU PRINCIPALE") }
    }
}

private fun compactNativeSummary(nativeLog: String): String {
    val lines = nativeLog.lineSequence().map { it.trim() }.filter { it.isNotBlank() }.toList()
    fun last(prefix: String): String? = lines.lastOrNull { it.startsWith(prefix) }
    val cacheCount = lines.count { it.startsWith("CACHE OK:") }
    val generationCount = lines.count { it.startsWith("GENERATE OK:") }
    val timeout = lines.lastOrNull { it.startsWith("TIMEOUT:") }
    return buildString {
        appendLine("Preparazioni cache registrate: $cacheCount")
        appendLine("Generazioni registrate: $generationCount")
        appendLine("Ultimo caricamento: ${last("LOAD OK:") ?: "non disponibile"}")
        appendLine("Ultima cache: ${last("CACHE OK:") ?: "non disponibile"}")
        appendLine("Ultima analisi: ${last("PROMPT OK:") ?: "non disponibile"}")
        appendLine("Ultimo primo token nativo: ${last("FIRST TOKEN:") ?: "non disponibile"}")
        append("Ultima generazione: ${last("GENERATE OK:") ?: "non disponibile"}")
        if (timeout != null) {
            appendLine()
            append("Ultimo timeout: $timeout")
        }
    }
}
