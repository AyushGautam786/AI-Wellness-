import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Send, Mic, ArrowLeft, Heart, Brain } from "lucide-react";
// import { emotionApi, EmotionPrediction } from "@/utils/emotionApiService";
// import StoryMappingService from "@/utils/emotionStoryMapping";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  emotion?: EmotionPrediction;
  isEmotionBased?: boolean;
}

const ChatInterface = ({ onBack }: { onBack: () => void }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      text: "Hello! I'm here to support you on your wellness journey. Feel free to share what's on your mind, or ask me about mindfulness exercises, journaling prompts, or coping strategies. How are you feeling today?",
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getAIResponse = (userMessage: string): string => {
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

    // Simulate AI thinking time
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: getAIResponse(inputValue),
        isUser: false,
        timestamp: new Date()
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
          
          <div>
            <h1 className="text-xl font-semibold text-foreground">Wellness Companion</h1>
            <p className="text-sm text-muted-foreground">Your safe space for emotional support</p>
          </div>
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
                <p className="leading-relaxed">{message.text}</p>
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
                  <span className="text-sm text-muted-foreground">AI is thinking...</span>
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
              placeholder="Share what's on your mind..."
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
          
          <p className="text-xs text-muted-foreground mt-2 text-center">
            This is a demo. For professional mental health support, please consult a qualified therapist.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;