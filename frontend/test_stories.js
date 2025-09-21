// Simple test script to verify therapeutic stories are working
import { DatasetStoryService } from './src/utils/datasetStoryService.js';

console.log('Testing DatasetStoryService...');

// Test getting stories for different emotions
const emotions = ['joy', 'sadness', 'love', 'anger', 'fear', 'surprise'];

emotions.forEach(emotion => {
    console.log(`\n=== Testing ${emotion.toUpperCase()} ===`);
    
    // Test getting random story
    const story = DatasetStoryService.getRandomStoryForEmotion(emotion);
    if (story) {
        console.log(`Story Title: ${story.title}`);
        console.log(`Story Moral: ${story.moral}`);
        console.log(`Follow-up Questions: ${story.follow_up_questions.length}`);
    } else {
        console.log(`No story found for ${emotion}`);
    }
    
    // Test therapeutic response
    const response = DatasetStoryService.generateTherapeuticResponse(emotion, 0.8, `I'm feeling ${emotion} today`);
    console.log('Therapeutic Response Preview:', response.substring(0, 100) + '...');
});

console.log('\n=== Test Complete ===');