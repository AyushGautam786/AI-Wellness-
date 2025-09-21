/**
 * Dataset Story Loader Service
 * Loads and manages therapeutic stories from the dataset JSON files
 */

export interface TherapeuticStory {
  id: string;
  title: string;
  content: string;
  moral: string;
  therapeutic_elements: string[];
  follow_up_questions: string[];
}

export interface EmotionDataset {
  emotion_description: string;
  therapeutic_goal: string;
  stories: TherapeuticStory[];
}

// Since we can't directly import JSON in this context, we'll provide the stories as constants
// In a real application, these would be loaded from the JSON files

export const THERAPEUTIC_STORIES: Record<string, EmotionDataset> = {
  joy: {
    emotion_description: "Happiness, celebration, contentment, and positive well-being",
    therapeutic_goal: "Amplify positive emotions, practice gratitude, maintain emotional balance",
    stories: [
      {
        id: "joy_001",
        title: "The Gratitude Garden",
        content: "Maya had always rushed through her daily walk to work, mind busy with endless tasks. One morning, running late, she was forced to slow down when construction blocked her usual path. Taking an unfamiliar route through a small neighborhood, she noticed something wonderful: tiny gardens tucked between houses, each one lovingly tended. An elderly man watering his roses smiled and waved. A child's chalk art decorated the sidewalk. A little free library invited her to take a book. By the time Maya reached work, she realized her detour had filled her with more joy than any efficient commute ever had. From that day forward, she chose the longer, more beautiful route, understanding that rushing past life's small wonders was the real waste of time.",
        moral: "Joy often hides in the moments we usually rush past; slowing down reveals everyday magic",
        therapeutic_elements: ["mindfulness", "gratitude practice", "present moment awareness"],
        follow_up_questions: [
          "What small daily moments could you slow down to appreciate more fully?",
          "How might changing your routine reveal new sources of joy?",
          "What everyday 'detours' might actually enrich your life rather than delay it?"
        ]
      },
      {
        id: "joy_002", 
        title: "The Shared Celebration",
        content: "When Kenji got the promotion he'd worked toward for years, his first instinct was to celebrate quietly - he worried about making others feel bad or seeming boastful. But his friend Anna convinced him to have a small party. As friends gathered to congratulate him, something beautiful happened: everyone began sharing their own recent victories, both big and small. Sarah had finally learned to bake bread, Marcus had reconnected with an old friend, and Lin had completed her first 5K run. The evening became a celebration not just of Kenji's promotion, but of all the ways they were each growing and succeeding. Kenji learned that joy shared doesn't diminish - it multiplies, creating a circle of mutual celebration and support.",
        moral: "Sharing joy creates more joy; celebrating together strengthens bonds and amplifies happiness",
        therapeutic_elements: ["social connection", "vulnerability", "mutual support"],
        follow_up_questions: [
          "How comfortable are you with sharing your successes with others?",
          "Who in your life would genuinely celebrate your victories with you?",
          "What stops you from acknowledging and celebrating your achievements?"
        ]
      }
    ]
  },

  sadness: {
    emotion_description: "Grief, loss, melancholy, and emotional pain",
    therapeutic_goal: "Process difficult emotions, find meaning in struggle, build resilience",
    stories: [
      {
        id: "sadness_001",
        title: "The Weaver's Thread",
        content: "Elena's grandmother had taught her to weave, and they spent countless afternoons together at the old loom. When her grandmother passed away, Elena couldn't bear to touch the weaving - it held too many memories. Months later, while cleaning out the attic, she discovered an unfinished tapestry her grandmother had been working on. Through her tears, Elena saw that the pattern was incomplete, but beautiful. She realized her grandmother had left her not just the tools to finish it, but the skills and love to continue creating. Slowly, carefully, Elena began adding her own threads to the pattern. The tapestry would never be exactly what her grandmother envisioned, but it became something new - a blend of both their hands, their love continuing to create beauty together even after goodbye.",
        moral: "Love transcends loss; we can honor those we've lost by continuing to create and grow",
        therapeutic_elements: ["grief processing", "continuing bonds", "meaning-making"],
        follow_up_questions: [
          "How do you honor the memory of those you've lost while still moving forward?",
          "What gifts or lessons from loved ones do you carry with you?",
          "How might your sadness be connected to the depth of love you've experienced?"
        ]
      },
      {
        id: "sadness_002",
        title: "The Winter Tree",
        content: "David sat by the window during the darkest part of winter, watching a bare oak tree in his yard. Leafless and seemingly lifeless, it looked as sad as he felt after losing his job and struggling with depression. One particularly cold morning, he noticed something: the tree's branches were covered in intricate ice formations that caught the light like crystal artwork. Even in its apparent emptiness, the tree was creating something beautiful. As weeks passed, David began to see other signs of life - tiny buds forming, birds resting in its strong branches, its roots holding firm against winter storms. He realized the tree wasn't dead during winter; it was resting, gathering strength for spring. David began to wonder if his own sadness might also be a season - not an ending, but a time of hidden preparation for whatever would come next.",
        moral: "Sadness can be a season of rest and preparation; even in apparent emptiness, life persists",
        therapeutic_elements: ["hope building", "seasonal metaphor", "resilience"],
        follow_up_questions: [
          "What signs of life or strength do you notice in yourself even during difficult times?",
          "How might this sadness be preparing you for future growth?",
          "What would it mean to trust that this is a season rather than a permanent state?"
        ]
      }
    ]
  },

  love: {
    emotion_description: "Deep affection, connection, and care for others or oneself",
    therapeutic_goal: "Build healthy relationships, practice self-love, strengthen connections",
    stories: [
      {
        id: "love_001",
        title: "The Two Gardens",
        content: "Alex and Jordan were neighbors who each tended beautiful gardens. Alex's was perfectly manicured, every plant in its designated spot. Jordan's was wild and free, with flowers spilling over paths and vegetables growing wherever they pleased. They often looked over the fence, each thinking the other's garden was better. One day, a storm damaged both gardens. Working together to clean up, they realized something beautiful: Alex's precision had created the perfect framework for Jordan's wild creativity, while Jordan's abundance had inspired Alex to let beauty grow beyond boundaries. They decided to remove the fence. Their combined garden became more magnificent than either had been alone.",
        moral: "Love thrives when different strengths complement each other rather than compete",
        therapeutic_elements: ["relationship dynamics", "complementarity", "partnership"],
        follow_up_questions: [
          "How do your unique qualities complement those you love?",
          "What boundaries in your life might benefit from being softened or removed?",
          "When have you created something beautiful by combining different approaches or perspectives?"
        ]
      }
    ]
  },

  anger: {
    emotion_description: "Intense emotional response to perceived injustice, threat, or frustration",
    therapeutic_goal: "Validate anger, channel it constructively, build healthy boundaries",
    stories: [
      {
        id: "anger_001",
        title: "The Blacksmith's Fire",
        content: "Young Thomas was apprenticed to a master blacksmith who seemed impossible to please. Every mistake brought harsh criticism, every success was ignored. Thomas's anger burned hotter each day until he was ready to quit. One evening, he watched the master work, hammering red-hot iron. 'Why do you heat it so much?' Thomas asked bitterly. 'Because,' the blacksmith replied, not looking up, 'cold iron breaks when you try to shape it. But iron heated with the right fire becomes stronger than it ever was before. Your anger is the fire, Thomas. The question is: will you let it destroy you, or will you use it to forge yourself into something unbreakable?' Thomas realized his anger wasn't his enemy - it was his forge.",
        moral: "Anger, when channeled correctly, can be the fire that shapes us into our strongest selves",
        therapeutic_elements: ["anger as energy", "transformation", "constructive channeling"],
        follow_up_questions: [
          "How might your anger be trying to forge you into something stronger?",
          "What injustices fuel your anger that you could work to change?",
          "How can you use anger's energy constructively rather than destructively?"
        ]
      }
    ]
  },

  fear: {
    emotion_description: "Protective response to perceived threat or uncertainty",
    therapeutic_goal: "Acknowledge fear's wisdom, build courage, develop coping strategies",
    stories: [
      {
        id: "fear_001",
        title: "The Cave Explorer's Light",
        content: "Maya stood at the entrance to the cave, her flashlight beam swallowed by darkness. Every instinct screamed 'danger,' but she knew the rare crystals she studied were inside. Her fear felt overwhelming until her mentor asked: 'What is fear trying to tell you?' Maya paused. 'To be careful. To bring enough light. To tell someone where I'm going.' Her mentor smiled. 'Then listen to that wisdom.' Maya prepared thoroughly - extra batteries, backup flashlight, detailed plan, check-in schedule. When she finally entered the cave, fear walked beside her not as an enemy, but as a wise counselor ensuring her safety. She found the crystals and returned safely, grateful for fear's protective guidance.",
        moral: "Fear is often wisdom disguised as worry, teaching us to prepare and proceed mindfully",
        therapeutic_elements: ["fear as wisdom", "preparation", "mindful courage"],
        follow_up_questions: [
          "What wisdom might your fears be offering about how to proceed safely?",
          "How can you honor fear's protective message while still moving forward?",
          "What preparation would help you feel more confident facing your fears?"
        ]
      }
    ]
  },

  surprise: {
    emotion_description: "Sudden awareness of unexpected changes or revelations",
    therapeutic_goal: "Embrace spontaneity, cultivate openness, find joy in unexpected moments",
    stories: [
      {
        id: "surprise_001",
        title: "The Unexpected Garden",
        content: "After months of careful planning, Lisa's vegetable garden was perfect - neat rows of beans, precisely spaced tomatoes, organized herb sections. Then the rabbits came. They ate the lettuce, scattered the mulch, and seemed to replant seeds randomly. Lisa was frustrated until she noticed something surprising: volunteer plants were sprouting everywhere. Pumpkins grew where she'd planted carrots, sunflowers appeared among the beans, and a wild patch of mixed greens was thriving in the corner. Her 'ruined' garden had become more abundant and beautiful than her planned one. Lisa learned that sometimes life's best gifts come not from our careful planning, but from our willingness to be surprised by what wants to grow.",
        moral: "Unexpected changes often bring gifts that our careful plans could never have imagined",
        therapeutic_elements: ["adaptability", "unexpected gifts", "letting go of control"],
        follow_up_questions: [
          "When has an unexpected change brought surprising benefits to your life?",
          "How might you be more open to life's spontaneous gifts?",
          "What would it feel like to trust that unexpected doesn't always mean unwelcome?"
        ]
      }
    ]
  }
};

export class DatasetStoryService {
  static getStoriesForEmotion(emotion: string): TherapeuticStory[] {
    const dataset = THERAPEUTIC_STORIES[emotion.toLowerCase()];
    return dataset ? dataset.stories : [];
  }

  static getRandomStoryForEmotion(emotion: string): TherapeuticStory | null {
    const stories = this.getStoriesForEmotion(emotion);
    if (stories.length === 0) return null;
    
    const randomIndex = Math.floor(Math.random() * stories.length);
    return stories[randomIndex];
  }

  static generateTherapeuticResponse(emotion: string, confidence: number, userMessage: string): string {
    const story = this.getRandomStoryForEmotion(emotion);
    
    if (!story) {
      return `I can sense you're feeling ${emotion}. While I don't have a specific story for this emotion right now, I want you to know that all feelings are valid and important. Would you like to tell me more about what's on your mind?`;
    }

    // Create a personalized response with the therapeutic story
    const response = `I can sense the ${emotion} in your words. Let me share a therapeutic story that might resonate with what you're experiencing:

**${story.title}**

${story.content}

${story.moral}

I hope this story offers you some comfort and perspective. Remember, ${emotion} is a natural and valid emotion that carries its own wisdom.`;

    return response;
  }

  static getFollowUpQuestions(emotion: string): string[] {
    const story = this.getRandomStoryForEmotion(emotion);
    return story ? story.follow_up_questions : [
      "How are you feeling right now?",
      "What would be most helpful for you in this moment?",
      "Would you like to explore this feeling further?"
    ];
  }

  static getTherapeuticGoal(emotion: string): string {
    const dataset = THERAPEUTIC_STORIES[emotion.toLowerCase()];
    return dataset ? dataset.therapeutic_goal : "Provide emotional support and understanding";
  }
}

export default DatasetStoryService;