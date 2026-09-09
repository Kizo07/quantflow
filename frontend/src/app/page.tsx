import { DigitalRain } from "@/components/landing/digital-rain";
import { Footer } from "@/components/landing/footer";
import { Header } from "@/components/landing/header";
import { Hero } from "@/components/landing/hero";
import { CaseStudySection } from "@/components/landing/sections/case-study-section";
import { CommunitySection } from "@/components/landing/sections/community-section";
import { SandboxSection } from "@/components/landing/sections/sandbox-section";
import { SkillsSection } from "@/components/landing/sections/skills-section";
import { WhatsNewSection } from "@/components/landing/sections/whats-new-section";
import { DEFAULT_LOCALE } from "@/core/i18n/locale";

export default function LandingPage() {
  return (
    <div className="relative min-h-screen w-full overflow-x-clip bg-[#020609]">
      <DigitalRain />
      <Header locale={DEFAULT_LOCALE} />
      <main className="relative z-10 flex w-full flex-col">
        <Hero />
        <CaseStudySection />
        <SkillsSection />
        <SandboxSection />
        <WhatsNewSection />
        <CommunitySection />
      </main>
      <Footer className="relative z-10" />
    </div>
  );
}
