import { Button } from "@/components/ui/button";
import heroImage from "@/assets/hero-wellness.jpg";

const HeroSection = ({ onStartChat }: { onStartChat: () => void }) => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
      {/* Background with gradient overlay */}
      <div className="absolute inset-0 bg-gradient-hero"></div>
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-30"
        style={{ backgroundImage: `url(${heroImage})` }}
      ></div>
      
      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 text-center text-white">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
            Your AI-Powered
            <span className="block bg-gradient-to-r from-white to-primary-glow bg-clip-text text-transparent">
              Wellness Companion
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl mb-8 text-gray-100 leading-relaxed">
            Find peace, clarity, and emotional support through empathetic AI conversations. 
            Get personalized mindfulness guidance and CBT-based tools for mental wellness.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button 
              onClick={onStartChat}
              size="lg" 
              className="bg-white text-primary hover:bg-gray-100 shadow-warm transition-smooth text-lg px-8 py-4 rounded-xl"
            >
              Start Your Wellness Journey
            </Button>
            
            <Button 
              variant="outline" 
              size="lg"
              className="border-white text-white hover:bg-white hover:text-primary transition-smooth text-lg px-8 py-4 rounded-xl"
            >
              Learn More
            </Button>
          </div>
        </div>
        
        {/* Features preview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 max-w-5xl mx-auto">
          {[
            {
              title: "Empathetic Conversations",
              description: "Talk openly about your feelings with an AI that truly listens and understands"
            },
            {
              title: "Mindfulness Guidance", 
              description: "Personalized meditation and breathing exercises tailored to your needs"
            },
            {
              title: "CBT Techniques",
              description: "Evidence-based cognitive behavioral therapy tools to reframe negative thoughts"
            }
          ].map((feature, index) => (
            <div 
              key={index}
              className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20 shadow-soft"
            >
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-gray-200">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HeroSection;