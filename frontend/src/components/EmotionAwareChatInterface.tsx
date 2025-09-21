import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Send, Mic, ArrowLeft, Heart, Brain, Zap } from "lucide-react";
import { DatasetStoryService } from "@/utils/datasetStoryService";

interface EmotionPrediction {
  predicted_emotion: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  success: boolean;
  error?: string;
}

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  emotion?: EmotionPrediction;
  isEmotionBased?: boolean;
  followUpQuestions?: string[];
}

const ChatInterface = ({ onBack }: { onBack: () => void }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      text: "Hello! I'm here to support you on your wellness journey. I can now understand your emotions and provide personalized stories to help you. Feel free to share what's on your mind - I'll sense how you're feeling and offer you a story that resonates with your current emotional state. How are you feeling today?",
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isEmotionDetectionEnabled, setIsEmotionDetectionEnabled] = useState(true);
  const [apiStatus, setApiStatus] = useState<'checking' | 'available' | 'unavailable'>('checking');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check emotion API availability on component mount
  useEffect(() => {
    checkEmotionApiStatus();
  }, []);

  const checkEmotionApiStatus = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/health');
      if (response.ok) {
        const data = await response.json();
        setApiStatus(data.model_loaded ? 'available' : 'unavailable');
      } else {
        setApiStatus('unavailable');
      }
    } catch (error) {
      console.log('Emotion API unavailable:', error);
      setApiStatus('unavailable');
    }
  };

  const predictEmotion = async (text: string): Promise<EmotionPrediction | null> => {
    if (apiStatus !== 'available' || !isEmotionDetectionEnabled) {
      return null;
    }

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (response.ok) {
        const data = await response.json();
        return data;
      }
    } catch (error) {
      console.error('Emotion prediction error:', error);
    }
    return null;
  };

  const getEmotionBasedResponse = (emotion: string, confidence: number, userMessage: string): string => {
    // Use the DatasetStoryService to generate a proper therapeutic story response
    const storyResponse = DatasetStoryService.generateTherapeuticResponse(emotion, confidence, userMessage);
    
    // Add confidence indicator if high confidence
    if (confidence > 0.7) {
      return `(Emotion detected: ${emotion} - ${Math.round(confidence * 100)}% confidence)\n\n${storyResponse}`;
    }
    
    return storyResponse;
  };

  const getFallbackResponse = (userMessage: string): string => {
    const responses = [
      "I understand how you're feeling. It's completely normal to experience these emotions. Let's explore this together - what specific aspect of this situation feels most challenging for you?",
      
      "Thank you for sharing that with me. It takes courage to open up about difficult feelings. Have you noticed any physical sensations that accompany these thoughts? Sometimes our body gives us important signals about our emotional state.",
      
      "That sounds really difficult. I want you to know that your feelings are valid. Let's try a quick grounding exercise: Can you tell me 5 things you can see around you right now, 4 things you can touch, and 3 things you can hear?",
      
      "I hear you, and I'm here to support you. When thoughts like these arise, it can be helpful to ask ourselves: 'Is this thought helpful right now?' and 'What would I tell a dear friend experiencing the same thing?' What comes up for you when you consider these questions?",
      
      "Your awareness of these feelings shows great emotional intelligence. Let's work on reframing this situation. Instead of focusing on what went wrong, can you think of one small thing that you handled well today, no matter how minor it might seem?"
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true);

    // Try to predict emotion
    const emotion = await predictEmotion(inputValue);

    // Simulate AI thinking time
    setTimeout(() => {
      let responseText: string;
      let isEmotionBased = false;
      let followUpQuestions: string[] = [];

      if (emotion && emotion.success) {
        responseText = getEmotionBasedResponse(emotion.predicted_emotion, emotion.confidence, inputValue);
        followUpQuestions = DatasetStoryService.getFollowUpQuestions(emotion.predicted_emotion);
        isEmotionBased = true;
      } else {
        responseText = getFallbackResponse(inputValue);
      }

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: responseText,
        isUser: false,
        timestamp: new Date(),
        emotion: emotion || undefined,
        isEmotionBased,
        followUpQuestions
      };
      
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1500 + Math.random() * 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getEmotionIcon = (emotion: string) => {
    const icons = {
      joy: "😊",
      sadness: "😢",
      love: "❤️",
      anger: "😠",
      fear: "😰",
      surprise: "😲"
    };
    return icons[emotion as keyof typeof icons] || "🎭";
  };

  const getEmotionColor = (emotion: string) => {
    const colors = {
      joy: "text-yellow-600",
      sadness: "text-blue-600",
      love: "text-pink-600",
      anger: "text-red-600",
      fear: "text-purple-600",
      surprise: "text-orange-600"
    };
    return colors[emotion as keyof typeof colors] || "text-gray-600";
  };

  return (
    <div className="min-h-screen bg-gradient-secondary flex flex-col">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-border p-4 shadow-soft">
        <div className="container mx-auto flex items-center gap-4">
          <Button 
            onClick={onBack}
            variant="ghost" 
            size="icon"
            className="hover:bg-muted"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          
          <div className="flex-1">
            <h1 className="text-xl font-semibold text-foreground flex items-center gap-2">
              Wellness Companion
              {apiStatus === 'available' && (
                <span className="text-sm bg-green-100 text-green-800 px-2 py-1 rounded-full flex items-center gap-1">
                  <Brain className="h-3 w-3" />
                  Emotion AI
                </span>
              )}
            </h1>
            <p className="text-sm text-muted-foreground">
              {apiStatus === 'available' 
                ? "Your safe space with emotion-aware AI support" 
                : "Your safe space for emotional support"
              }
            </p>
          </div>

          {apiStatus === 'available' && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsEmotionDetectionEnabled(!isEmotionDetectionEnabled)}
              className={`${isEmotionDetectionEnabled ? 'bg-blue-50 text-blue-700' : ''}`}
            >
              <Heart className="h-4 w-4 mr-1" />
              {isEmotionDetectionEnabled ? 'Emotion AI On' : 'Emotion AI Off'}
            </Button>
          )}
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="container mx-auto max-w-4xl space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}
            >
              <Card className={`max-w-[70%] p-4 shadow-soft ${
                message.isUser 
                  ? "bg-primary text-primary-foreground" 
                  : "bg-card border-border"
              }`}>
                <p className="leading-relaxed whitespace-pre-wrap">{message.text}</p>
                
                {/* Follow-up questions for therapeutic stories */}
                {!message.isUser && message.followUpQuestions && message.followUpQuestions.length > 0 && (
                  <div className="mt-4 p-3 bg-muted/30 rounded-lg border-l-4 border-primary/30">
                    <p className="text-sm font-medium text-muted-foreground mb-2">💭 Reflection Questions:</p>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      {message.followUpQuestions.map((question, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-primary mt-1">•</span>
                          <span>{question}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Emotion indicator for AI responses */}
                {!message.isUser && message.emotion && message.emotion.success && (
                  <div className="mt-3 pt-3 border-t border-border/30">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Zap className="h-3 w-3" />
                      <span>Detected emotion:</span>
                      <span className={`font-medium ${getEmotionColor(message.emotion.predicted_emotion)}`}>
                        {getEmotionIcon(message.emotion.predicted_emotion)} {message.emotion.predicted_emotion}
                      </span>
                      <span>({Math.round(message.emotion.confidence * 100)}%)</span>
                    </div>
                  </div>
                )}

                <p className={`text-xs mt-2 ${
                  message.isUser 
                    ? "text-primary-foreground/70" 
                    : "text-muted-foreground"
                }`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </Card>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex justify-start">
              <Card className="bg-card border-border p-4 shadow-soft">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse" style={{ animationDelay: "0.2s" }}></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse" style={{ animationDelay: "0.4s" }}></div>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {apiStatus === 'available' ? 'AI is analyzing your emotions and crafting a response...' : 'AI is thinking...'}
                  </span>
                </div>
              </Card>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-border bg-white/80 backdrop-blur-sm p-4">
        <div className="container mx-auto max-w-4xl">
          <div className="flex items-center space-x-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                apiStatus === 'available' 
                  ? "Share your thoughts and feelings... I'll understand your emotions ✨" 
                  : "Share what's on your mind..."
              }
              className="flex-1 bg-background border-border shadow-soft"
              disabled={isTyping}
            />
            
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isTyping}
              className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-soft"
            >
              <Send className="h-4 w-4" />
            </Button>
            
            <Button
              variant="outline"
              className="border-border hover:bg-muted"
              disabled
              title="Voice input (coming soon)"
            >
              <Mic className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="text-xs text-muted-foreground mt-2 text-center">
            {apiStatus === 'available' ? (
              <span className="flex items-center justify-center gap-1">
                <Brain className="h-3 w-3" />
                Emotion AI is active - your feelings will be understood and responded to with care
              </span>
            ) : apiStatus === 'unavailable' ? (
              <span>Emotion AI unavailable - using standard wellness support mode</span>
            ) : (
              <span>Checking emotion AI availability...</span>
            )}
            <br />
            This is a demo. For professional mental health support, please consult a qualified therapist.
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;