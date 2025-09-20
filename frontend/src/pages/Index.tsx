import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import ServicesSection from "@/components/ServicesSection";
import TestimonialsSection from "@/components/TestimonialsSection";
import EmotionAwareChatInterface from "@/components/EmotionAwareChatInterface";

const Index = () => {
  const [showChat, setShowChat] = useState(false);
  const navigate = useNavigate();

  const handleStartChat = () => {
    setShowChat(true);
  };

  const handleBackToHome = () => {
    setShowChat(false);
  };

  const handleLoginClick = () => {
    navigate("/login");
  };

  const handleRegisterClick = () => {
    navigate("/register");
  };

  if (showChat) {
    return <EmotionAwareChatInterface onBack={handleBackToHome} />;
  }

  return (
    <div className="min-h-screen">
      <Header onLoginClick={handleLoginClick} onRegisterClick={handleRegisterClick} />
      
      <main>
        <div id="home">
          <HeroSection onStartChat={handleStartChat} />
        </div>
        
        <div id="about">
          <ServicesSection />
        </div>
        
        <div id="testimonials">
          <TestimonialsSection />
        </div>
        
        {/* Contact Section */}
        <section id="contact" className="py-20 bg-background border-t border-border">
          <div className="container mx-auto px-6 text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
              Ready to Start Your Journey?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Take the first step toward better mental wellness today. Our AI companion is here to support you every step of the way.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={handleStartChat}
                className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-4 rounded-xl font-semibold text-lg shadow-soft transition-smooth"
              >
                Start Free Conversation
              </button>
              <button
                onClick={handleRegisterClick}
                className="border border-primary text-primary hover:bg-primary hover:text-primary-foreground px-8 py-4 rounded-xl font-semibold text-lg transition-smooth"
              >
                Create Account
              </button>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-muted py-12 border-t border-border">
          <div className="container mx-auto px-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              <div className="md:col-span-2">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-primary rounded-lg flex items-center justify-center shadow-soft">
                    <span className="text-white font-bold">❤️</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-foreground">MindfulAI</h3>
                    <p className="text-sm text-muted-foreground">Wellness Companion</p>
                  </div>
                </div>
                <p className="text-muted-foreground mb-4">
                  Empowering mental wellness through AI-powered conversations, mindfulness guidance, and evidence-based therapeutic techniques.
                </p>
                <p className="text-sm text-muted-foreground">
                  © 2024 MindfulAI. All rights reserved.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-foreground mb-4">Support</h4>
                <ul className="space-y-2 text-muted-foreground">
                  <li><a href="#" className="hover:text-primary transition-smooth">Help Center</a></li>
                  <li><a href="#" className="hover:text-primary transition-smooth">Privacy Policy</a></li>
                  <li><a href="#" className="hover:text-primary transition-smooth">Terms of Service</a></li>
                  <li><a href="#" className="hover:text-primary transition-smooth">Contact Us</a></li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-foreground mb-4">Resources</h4>
                <ul className="space-y-2 text-muted-foreground">
                  <li><a href="#" className="hover:text-primary transition-smooth">Mental Health Blog</a></li>
                  <li><a href="#" className="hover:text-primary transition-smooth">Wellness Tips</a></li>
                  <li><a href="#" className="hover:text-primary transition-smooth">Crisis Resources</a></li>
                  <li><a href="#" className="hover:text-primary transition-smooth">Professional Help</a></li>
                </ul>
              </div>
            </div>
            
            <div className="mt-8 pt-8 border-t border-border text-center text-sm text-muted-foreground">
              <p>
                <strong>Disclaimer:</strong> MindfulAI is not a replacement for professional mental health treatment. 
                If you're experiencing a mental health crisis, please contact a qualified professional or emergency services.
              </p>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
};

export default Index;