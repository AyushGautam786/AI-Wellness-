import { Card } from "@/components/ui/card";
import { Star } from "lucide-react";

const TestimonialsSection = () => {
  const testimonials = [
    {
      id: 1,
      name: "Sarah Johnson",
      role: "Marketing Professional",
      content: "MindfulAI has been a game-changer for my mental health. The personalized mindfulness exercises and empathetic conversations have helped me manage stress and anxiety better than ever before.",
      rating: 5,
      avatar: "SJ"
    },
    {
      id: 2,
      name: "Michael Chen",
      role: "Software Developer",
      content: "As someone who struggles with overthinking, the CBT techniques provided by this AI companion have been incredibly helpful. It's like having a therapist available 24/7.",
      rating: 5,
      avatar: "MC"
    },
    {
      id: 3,
      name: "Emily Rodriguez",
      role: "Teacher",
      content: "The journaling prompts and personalized stories have helped me process difficult emotions and develop better coping mechanisms. Highly recommended!",
      rating: 5,
      avatar: "ER"
    },
    {
      id: 4,
      name: "David Thompson",
      role: "Healthcare Worker",
      content: "Working in a high-stress environment, I needed something accessible and immediate. MindfulAI provides exactly that - instant support whenever I need it most.",
      rating: 5,
      avatar: "DT"
    },
    {
      id: 5,
      name: "Lisa Park",
      role: "Student",
      content: "The AI's empathetic responses feel so genuine and helpful. It's helped me through exam stress and relationship challenges with practical, evidence-based advice.",
      rating: 5,
      avatar: "LP"
    },
    {
      id: 6,
      name: "James Wilson",
      role: "Entrepreneur",
      content: "I was skeptical at first, but the personalized approach and consistent availability have made this an essential part of my mental wellness routine.",
      rating: 5,
      avatar: "JW"
    }
  ];

  return (
    <section id="testimonials" className="py-20 bg-gradient-secondary">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
            What Our Users Say
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Join thousands of people who have transformed their mental wellness journey with MindfulAI
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial) => (
            <Card key={testimonial.id} className="p-6 shadow-soft hover:shadow-warm transition-smooth bg-card border-border">
              {/* Rating Stars */}
              <div className="flex items-center mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="h-5 w-5 fill-primary text-primary" />
                ))}
              </div>

              {/* Testimonial Content */}
              <p className="text-foreground mb-6 leading-relaxed">
                "{testimonial.content}"
              </p>

              {/* User Info */}
              <div className="flex items-center">
                <div className="w-12 h-12 bg-gradient-primary rounded-full flex items-center justify-center text-white font-semibold shadow-soft mr-4">
                  {testimonial.avatar}
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">{testimonial.name}</h4>
                  <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Trust Indicators */}
        <div className="text-center mt-16">
          <div className="flex flex-wrap justify-center items-center gap-8 text-muted-foreground">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-primary rounded-full"></div>
              <span className="font-medium">10,000+ Active Users</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-primary rounded-full"></div>
              <span className="font-medium">4.9/5 Average Rating</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-primary rounded-full"></div>
              <span className="font-medium">24/7 AI Support</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TestimonialsSection;