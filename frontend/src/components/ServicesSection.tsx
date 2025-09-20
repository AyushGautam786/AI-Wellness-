import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MessageCircle, Brain, BookOpen, Lightbulb, Shield, Clock } from "lucide-react";

const ServicesSection = () => {
  const services = [
    {
      icon: MessageCircle,
      title: "Empathetic Conversations",
      description: "Engage in meaningful dialogues with our AI that understands your emotions and provides compassionate support.",
      features: ["Active listening", "Emotional validation", "Safe space to share"]
    },
    {
      icon: Brain,
      title: "CBT Techniques",
      description: "Evidence-based Cognitive Behavioral Therapy methods to help reframe negative thoughts and build resilience.",
      features: ["Thought challenging", "Behavioral insights", "Coping strategies"]
    },
    {
      icon: BookOpen,
      title: "Mindfulness & Meditation",
      description: "Personalized mindfulness exercises, breathing techniques, and guided meditations for inner peace.",
      features: ["Guided meditations", "Breathing exercises", "Mindfulness practices"]
    },
    {
      icon: Lightbulb,
      title: "Personalized Stories",
      description: "Custom therapeutic stories and scenarios that illustrate coping mechanisms and inspire positive change.",
      features: ["Therapeutic narratives", "Relatable scenarios", "Inspiring content"]
    },
    {
      icon: Shield,
      title: "Safe & Confidential",
      description: "Your privacy is our priority. All conversations are encrypted and completely confidential.",
      features: ["End-to-end encryption", "No data sharing", "Complete anonymity"]
    },
    {
      icon: Clock,
      title: "24/7 Availability",
      description: "Mental health support whenever you need it, day or night, without appointments or waiting times.",
      features: ["Instant access", "No scheduling needed", "Always available"]
    }
  ];

  return (
    <section id="services" className="py-20 bg-background">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
            Comprehensive Mental Wellness Support
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our AI-powered platform offers a complete suite of mental health tools designed to support your journey toward emotional well-being
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => {
            const Icon = service.icon;
            return (
              <Card key={index} className="p-8 shadow-soft hover:shadow-warm transition-smooth bg-card border-border group">
                <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mb-6 shadow-soft group-hover:scale-110 transition-smooth">
                  <Icon className="h-8 w-8 text-white" />
                </div>

                <h3 className="text-2xl font-bold text-foreground mb-4">
                  {service.title}
                </h3>

                <p className="text-muted-foreground mb-6 leading-relaxed">
                  {service.description}
                </p>

                <ul className="space-y-2 mb-6">
                  {service.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center text-sm text-muted-foreground">
                      <div className="w-2 h-2 bg-primary rounded-full mr-3"></div>
                      {feature}
                    </li>
                  ))}
                </ul>

                <Button 
                  variant="outline" 
                  className="w-full border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-smooth"
                >
                  Learn More
                </Button>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default ServicesSection;