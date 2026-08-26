package com.neontides.nativeapp.ai

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream

class ModelManager(private val context: Context) {
    private val prefs = context.getSharedPreferences("models", Context.MODE_PRIVATE)

    fun modelsDir(): File = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }

    fun listModels(): List<File> = modelsDir().listFiles()
        ?.filter { it.extension.equals("gguf", true) }
        ?.sortedBy { it.name } ?: emptyList()

    fun activeModelName(): String? = prefs.getString("active", null)

    fun activeModelFile(): File? {
        val name = activeModelName() ?: return null
        val file = File(modelsDir(), name)
        if (!file.exists()) {
            prefs.edit().remove("active").apply()
            return null
        }
        return file
    }

    fun setActive(file: File) {
        // Lo scaricamento viene eseguito da AiEngine.restart() su Dispatchers.IO.
        // Chiamare JNI qui dal thread UI può bloccare mentre il modello genera.
        prefs.edit().putString("active", file.name).apply()
    }

    fun deleteModel(file: File): Boolean {
        val wasActive = activeModelName() == file.name
        if (wasActive) {
            prefs.edit().remove("active").apply()
        }
        val removed = !file.exists() || file.delete()
        if (removed && file.name == ModelDownloadManager.MODEL_NAME) {
            ModelDownloadManager(context).markModelDeleted()
        }
        return removed
    }

    fun importModel(uri: Uri): File {
        val originalName = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }?.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val baseName = originalName?.takeIf { it.endsWith(".gguf", true) }
            ?: "model_${System.currentTimeMillis()}.gguf"
        val fileName = if (File(modelsDir(), baseName).exists()) {
            baseName.removeSuffix(".gguf") + "_${System.currentTimeMillis()}.gguf"
        } else baseName
        val dest = File(modelsDir(), fileName)
        try {
            context.contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Impossibile aprire il file" }
                FileOutputStream(dest).use { output ->
                    input.copyTo(output, 1024 * 1024)
                }
            }
            require(dest.length() > 1024 * 1024) { "Il file GGUF è vuoto o incompleto" }
        } catch (t: Throwable) {
            dest.delete()
            throw t
        }
        setActive(dest)
        return dest
    }

    fun recommendedDownloader(): ModelDownloadManager = ModelDownloadManager(context)
}
