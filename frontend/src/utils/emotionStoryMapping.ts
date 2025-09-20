/**
 * Emotion-to-Story Mapping System
 * Maps detected emotions to appropriate story themes, templates, and content.
 */

export interface StoryTemplate {
  id: string;
  emotion: string;
  theme: string;
  premise: string;
  characters: string[];
  setting: string;
  tone: string;
  messageTemplate: string;
  followUpQuestions: string[];
}

export interface EmotionMapping {
  emotion: string;
  storyCategories: string[];
  therapeuticApproach: string;
  supportiveElements: string[];
  templates: StoryTemplate[];
}

// Story templates for each emotion
export const EMOTION_STORY_MAPPINGS: Record<string, EmotionMapping> = {
  joy: {
    emotion: "joy",
    storyCategories: ["celebration", "achievement", "friendship", "adventure"],
    therapeuticApproach: "amplification and gratitude",
    supportiveElements: ["positive reinforcement", "gratitude practices", "sharing joy"],
    templates: [
      {
        id: "joy-1",
        emotion: "joy",
        theme: "Celebrating Achievement",
        premise: "A character who overcomes challenges and celebrates their success",
        characters: ["The Achiever", "Supportive Friend", "Wise Mentor"],
        setting: "A place of accomplishment",
        tone: "uplifting and celebratory",
        messageTemplate: "I can sense the joy in your words! Let me share a story about someone who, like you, experienced a wonderful moment of triumph. {storyContent} Your happiness reminds me that celebrating our victories, no matter how small, is so important for our wellbeing.",
        followUpQuestions: [
          "What specific part of this achievement brings you the most joy?",
          "Who would you most like to share this happiness with?",
          "How can you carry this positive energy forward?"
        ]
      },
      {
        id: "joy-2",
        emotion: "joy",
        theme: "Unexpected Wonder",
        premise: "A character who discovers beauty in unexpected places",
        characters: ["The Explorer", "Kind Stranger", "Nature Guide"],
        setting: "A surprising beautiful location",
        tone: "wonder-filled and inspiring",
        messageTemplate: "Your joy is contagious! Here's a story about finding magic in the unexpected. {storyContent} Just like you're experiencing now, sometimes the most beautiful moments come when we least expect them.",
        followUpQuestions: [
          "What unexpected joys have you discovered recently?",
          "How does this feeling change your perspective on your day?",
          "What would you like to explore next with this positive energy?"
        ]
      }
    ]
  },

  sadness: {
    emotion: "sadness",
    storyCategories: ["healing", "resilience", "comfort", "hope"],
    therapeuticApproach: "validation and gentle support",
    supportiveElements: ["emotional validation", "hope building", "gentle comfort"],
    templates: [
      {
        id: "sadness-1",
        emotion: "sadness",
        theme: "The Phoenix Within",
        premise: "A character who learns that sadness is part of the journey to renewal",
        characters: ["The Wanderer", "Wise Elder", "Gentle Healer"],
        setting: "A quiet garden of reflection",
        tone: "gentle and hopeful",
        messageTemplate: "I hear the sadness in your words, and I want you to know that what you're feeling is completely valid. Let me share a story about finding light in dark moments. {storyContent} Remember, even in sadness, you're not alone, and this feeling won't last forever.",
        followUpQuestions: [
          "What would comfort feel like for you right now?",
          "Can you think of a time when sadness eventually led to growth?",
          "What small step toward self-care could you take today?"
        ]
      },
      {
        id: "sadness-2",
        emotion: "sadness",
        theme: "The Gentle Rain",
        premise: "A character who learns that sadness, like rain, nourishes growth",
        characters: ["The Gardener", "Understanding Friend", "Patient Teacher"],
        setting: "A peaceful rainy afternoon",
        tone: "soothing and nurturing",
        messageTemplate: "Your sadness is acknowledged and held with care. Here's a gentle story about how even difficult emotions serve a purpose. {storyContent} Like rain nourishes the earth, your tears are part of your healing process.",
        followUpQuestions: [
          "What does your sadness need right now - rest, understanding, or something else?",
          "Who in your life offers you comfort during difficult times?",
          "What self-compassionate words would you offer to a friend feeling this way?"
        ]
      }
    ]
  },

  love: {
    emotion: "love",
    storyCategories: ["connection", "romance", "family", "self-love"],
    therapeuticApproach: "nurturing and appreciation",
    supportiveElements: ["relationship building", "gratitude", "connection"],
    templates: [
      {
        id: "love-1",
        emotion: "love",
        theme: "The Heart's Garden",
        premise: "A character who tends to the garden of relationships with care",
        characters: ["The Gardener of Hearts", "Beloved Friend", "Wise Grandmother"],
        setting: "A beautiful shared garden",
        tone: "warm and nurturing",
        messageTemplate: "The love you're expressing is beautiful and precious. Here's a story about the power of deep connection. {storyContent} Love like yours makes the world brighter - both for you and those fortunate enough to share in it.",
        followUpQuestions: [
          "What does this love inspire you to do or become?",
          "How do you like to express love to others?",
          "What are you most grateful for in this relationship?"
        ]
      },
      {
        id: "love-2",
        emotion: "love",
        theme: "The Mirror of Self-Love",
        premise: "A character who learns to love themselves as deeply as they love others",
        characters: ["The Self-Discoverer", "Inner Wisdom", "Compassionate Voice"],
        setting: "A peaceful meditation space",
        tone: "gentle and affirming",
        messageTemplate: "The love you feel is a reflection of your beautiful heart. Let me share a story about the importance of loving ourselves too. {storyContent} Remember that the love you give to others can also be a gift you give to yourself.",
        followUpQuestions: [
          "How can you show yourself the same love you give to others?",
          "What qualities about yourself are you learning to appreciate?",
          "How does feeling loved change your relationship with yourself?"
        ]
      }
    ]
  },

  anger: {
    emotion: "anger",
    storyCategories: ["justice", "transformation", "boundaries", "empowerment"],
    therapeuticApproach: "validation and redirection",
    supportiveElements: ["anger validation", "healthy expression", "boundary setting"],
    templates: [
      {
        id: "anger-1",
        emotion: "anger",
        theme: "The Fire That Builds",
        premise: "A character who transforms anger into positive action and change",
        characters: ["The Advocate", "Wise Counselor", "Change Maker"],
        setting: "A place where justice is sought",
        tone: "understanding and empowering",
        messageTemplate: "I can feel the intensity of your anger, and I want you to know that anger often signals that something important to you has been threatened or violated. Here's a story about channeling that fire constructively. {storyContent} Your anger is valid, and it can be a powerful force for positive change when directed thoughtfully.",
        followUpQuestions: [
          "What boundary or value of yours feels like it's been crossed?",
          "What would justice or resolution look like in this situation?",
          "How might you channel this energy into positive action?"
        ]
      },
      {
        id: "anger-2",
        emotion: "anger",
        theme: "The Volcano's Wisdom",
        premise: "A character who learns to understand the message their anger carries",
        characters: ["The Seeker", "Ancient Wisdom Keeper", "Patient Guide"],
        setting: "A mountain peak of clarity",
        tone: "validating and instructive",
        messageTemplate: "Your anger is telling you something important. Let me share a story about listening to what our strongest emotions are trying to teach us. {storyContent} Anger often protects what we value most - understanding its message can lead to powerful insights.",
        followUpQuestions: [
          "What is your anger trying to protect or defend?",
          "What would you need to feel heard and understood in this situation?",
          "How can you honor your anger while also taking care of yourself?"
        ]
      }
    ]
  },

  fear: {
    emotion: "fear",
    storyCategories: ["courage", "overcoming", "safety", "growth"],
    therapeuticApproach: "reassurance and gradual empowerment",
    supportiveElements: ["safety building", "courage development", "gradual exposure"],
    templates: [
      {
        id: "fear-1",
        emotion: "fear",
        theme: "The Brave Heart",
        premise: "A character who discovers that courage isn't the absence of fear, but action despite it",
        characters: ["The Reluctant Hero", "Encouraging Guide", "Wise Protector"],
        setting: "A threshold between safety and unknown",
        tone: "reassuring and empowering",
        messageTemplate: "I recognize the fear you're experiencing, and I want you to know that feeling scared doesn't make you weak - it makes you human. Here's a story about finding courage in the midst of uncertainty. {storyContent} Remember, brave people feel fear too; they just don't let it stop them from moving forward.",
        followUpQuestions: [
          "What would you do if you knew you couldn't fail?",
          "What small step toward courage could you take today?",
          "Who or what helps you feel safer when you're afraid?"
        ]
      },
      {
        id: "fear-2",
        emotion: "fear",
        theme: "The Lighthouse in the Storm",
        premise: "A character who finds guidance and safety even in the scariest moments",
        characters: ["The Lost Traveler", "Lighthouse Keeper", "Storm Survivor"],
        setting: "A safe harbor during a storm",
        tone: "comforting and stabilizing",
        messageTemplate: "Your fear is understandable, and you don't have to face it alone. Let me tell you about finding beacons of safety in uncertain times. {storyContent} Even in the stormiest weather, there are lighthouses to guide us home.",
        followUpQuestions: [
          "What helps you feel most safe and grounded?",
          "What would you tell someone else who was feeling this same fear?",
          "What resources or support do you have available to you right now?"
        ]
      }
    ]
  },

  surprise: {
    emotion: "surprise",
    storyCategories: ["discovery", "adventure", "wonder", "growth"],
    therapeuticApproach: "exploration and curiosity",
    supportiveElements: ["curiosity encouragement", "adaptability", "openness"],
    templates: [
      {
        id: "surprise-1",
        emotion: "surprise",
        theme: "The Unexpected Gift",
        premise: "A character who learns that surprises often bring hidden opportunities",
        characters: ["The Curious Explorer", "Mysterious Benefactor", "Adaptive Learner"],
        setting: "A place of unexpected discovery",
        tone: "wonder-filled and curious",
        messageTemplate: "I can sense your surprise! Life has a way of catching us off guard, doesn't it? Here's a story about embracing the unexpected. {storyContent} Sometimes the most surprising moments lead to the most beautiful discoveries about ourselves and our world.",
        followUpQuestions: [
          "What opportunities might this surprise be revealing?",
          "How do you typically handle unexpected changes?",
          "What new perspective might this experience offer you?"
        ]
      },
      {
        id: "surprise-2",
        emotion: "surprise",
        theme: "The Plot Twist",
        premise: "A character who learns to dance with life's unexpected turns",
        characters: ["The Flexible Dancer", "Life's Choreographer", "Wise Improviser"],
        setting: "A stage where anything can happen",
        tone: "playful and adaptable",
        messageTemplate: "Surprise! Life just wrote a plot twist in your story. Let me share a tale about rolling with life's unexpected moments. {storyContent} The most interesting stories are often the ones with the most surprising turns.",
        followUpQuestions: [
          "What new possibilities does this surprise open up?",
          "How might you adapt and make the best of this unexpected situation?",
          "What would it look like to embrace this surprise with curiosity rather than resistance?"
        ]
      }
    ]
  }
};

// Helper functions for story generation
export class StoryMappingService {
  static getStoriesForEmotion(emotion: string): StoryTemplate[] {
    const mapping = EMOTION_STORY_MAPPINGS[emotion.toLowerCase()];
    return mapping ? mapping.templates : [];
  }

  static getRandomStoryForEmotion(emotion: string): StoryTemplate | null {
    const stories = this.getStoriesForEmotion(emotion);
    if (stories.length === 0) return null;
    
    const randomIndex = Math.floor(Math.random() * stories.length);
    return stories[randomIndex];
  }

  static getTherapeuticApproach(emotion: string): string {
    const mapping = EMOTION_STORY_MAPPINGS[emotion.toLowerCase()];
    return mapping ? mapping.therapeuticApproach : "supportive listening";
  }

  static getSupportiveElements(emotion: string): string[] {
    const mapping = EMOTION_STORY_MAPPINGS[emotion.toLowerCase()];
    return mapping ? mapping.supportiveElements : ["emotional support"];
  }

  static generateStoryResponse(emotion: string, confidence: number, userMessage: string): string {
    const story = this.getRandomStoryForEmotion(emotion);
    
    if (!story) {
      return `I can sense you're feeling ${emotion}. While I don't have a specific story for this emotion right now, I want you to know that all feelings are valid and temporary. Would you like to tell me more about what's on your mind?`;
    }

    // Generate contextual story content based on the template
    const storyContent = this.generateStoryContent(story, userMessage);
    
    // Replace the placeholder in the message template
    const response = story.messageTemplate.replace('{storyContent}', storyContent);
    
    return response;
  }

  private static generateStoryContent(template: StoryTemplate, userMessage: string): string {
    // This is a simplified story generator - in a full implementation,
    // you might use AI to generate more personalized stories
    const stories = {
      "joy-1": "There once was a person who worked tirelessly toward their goal. When they finally achieved it, they realized that the journey itself had made them stronger. They celebrated not just the destination, but every step that led them there.",
      
      "joy-2": "A traveler was walking a familiar path when they noticed something they'd never seen before - a small garden hidden behind an old wall. The beauty took their breath away, reminding them that wonder exists everywhere if we keep our eyes open.",
      
      "sadness-1": "In a quiet garden, a person sat with their tears, feeling the weight of loss. An old gardener approached and said, 'Even the strongest trees bend in the storm, but their roots grow deeper with each wind.' The person learned that sadness, too, could deepen their capacity for joy.",
      
      "sadness-2": "During a gentle rain, someone watched droplets nourish the earth. They realized their tears were like that rain - painful but necessary, washing away what no longer served and making space for new growth.",
      
      "love-1": "Two friends planted a garden together, each contributing different seeds. As it grew, they realized their love had created something more beautiful than either could have made alone - a living testament to the power of caring connection.",
      
      "love-2": "A person looked in the mirror and, for the first time, spoke to their reflection with the same kindness they showed their dearest friend. They discovered that self-love wasn't selfish - it was the foundation for loving others.",
      
      "anger-1": "When someone's rights were violated, they felt fire in their chest. Instead of lashing out, they channeled that fire into advocacy, becoming a voice for change. Their anger became a force for justice.",
      
      "anger-2": "A person felt volcanic anger rising within them. A wise teacher said, 'Your anger is a messenger - what is it trying to tell you?' They learned to listen to their anger's wisdom without being consumed by its flames.",
      
      "fear-1": "Standing at the edge of something new, a person's heart raced with fear. But they took one small step, then another. They discovered that courage wasn't about not being afraid - it was about moving forward anyway.",
      
      "fear-2": "Lost in a storm, a traveler spotted a lighthouse beam cutting through the darkness. They realized that even in their fear, there were always guides and safe harbors waiting to be found.",
      
      "surprise-1": "Life handed someone an unexpected package. Though they hadn't ordered it, when they opened it, they found exactly what they needed for their next chapter. Sometimes the universe knows what we need before we do.",
      
      "surprise-2": "A dancer was performing a practiced routine when the music suddenly changed. Instead of stopping, they began to improvise, creating something more beautiful than they had ever planned."
    };

    return stories[template.id] || "Here's a story about resilience, growth, and the beautiful complexity of human emotion...";
  }

  static getFollowUpQuestions(emotion: string): string[] {
    const story = this.getRandomStoryForEmotion(emotion);
    return story ? story.followUpQuestions : [
      "How are you feeling right now?",
      "What would be most helpful for you in this moment?",
      "Would you like to explore this feeling further?"
    ];
  }
}

export default StoryMappingService;