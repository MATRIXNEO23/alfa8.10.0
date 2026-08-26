package com.neontides.nativeapp.ai

import com.neontides.nativeapp.model.CharacterProfile
import com.neontides.nativeapp.model.Relationship
import org.junit.Assert.assertEquals
import org.junit.Test

class NaturalRelationshipScorerTest {
    private val luna = CharacterProfile(
        id = "luna",
        name = "Luna Hayashi",
        age = 22,
        job = "Cantautrice",
        personality = "Spontanea e creativa.",
        likes = listOf("musica rock", "notte", "libertà"),
        dislikes = listOf("routine", "controllo"),
        sensuality = 4
    )

    @Test
    fun ordinaryInterestAndDisclosureIncreaseTrustGradually() {
        assertEquals(1, NaturalRelationshipScorer.assess("Che musica ascolti?", luna, Relationship()).delta.trust)
        assertEquals(1, NaturalRelationshipScorer.assess("Mi chiamo Alberto e creo applicazioni", luna, Relationship()).delta.trust)
        assertEquals(0, NaturalRelationshipScorer.assess("ciao", luna, Relationship()).delta.trust)
    }

    @Test
    fun complimentsDoNotCreateImmediateSexualAttractionBetweenStrangers() {
        val stranger = NaturalRelationshipScorer.assess("Sei davvero affascinante", luna, Relationship()).delta
        assertEquals(1, stranger.affection)
        assertEquals(0, stranger.attraction)

        val established = NaturalRelationshipScorer.assess(
            "Sei davvero affascinante",
            luna,
            Relationship(stage = "Amicizia")
        ).delta
        assertEquals(1, established.attraction)
    }

    @Test
    fun sharedAndOpposedPreferencesHaveOppositeEffects() {
        val shared = NaturalRelationshipScorer.assess("Anche a me piace la musica rock", luna, Relationship()).delta
        assertEquals(1, shared.affection)
        assertEquals(1, shared.trust)

        val opposed = NaturalRelationshipScorer.assess("Odio la musica rock", luna, Relationship()).delta
        assertEquals(-1, opposed.affection)
        assertEquals(-1, opposed.trust)
    }
}
