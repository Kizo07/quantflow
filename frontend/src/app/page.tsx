import { Footer } from "@/components/landing/footer";
import { Header } from "@/components/landing/header";
import { Hero } from "@/components/landing/hero";
import { AlphaEngineSection } from "@/components/landing/sections/alpha-engine-section";
import { CaseStudySection } from "@/components/landing/sections/case-study-section";
import { CommunitySection } from "@/components/landing/sections/community-section";
import { KizoNLPSection } from "@/components/landing/sections/kizonlp-section";
import { KnowledgeSection } from "@/components/landing/sections/knowledge-section";
import { QuantDeskSection } from "@/components/landing/sections/quant-desk-section";
import { ReportForgeSection } from "@/components/landing/sections/report-forge-section";
import { SandboxSection } from "@/components/landing/sections/sandbox-section";
import { SkillsSection } from "@/components/landing/sections/skills-section";
import { SparkSection } from "@/components/landing/sections/spark-section";
import { DEFAULT_LOCALE } from "@/core/i18n/locale";

export default function LandingPage() {
  return (
    <div className="bg-km-bg min-h-screen w-full overflow-x-clip">
      <Header locale={DEFAULT_LOCALE} />
      <main className="flex w-full flex-col">
        <Hero />
        <QuantDeskSection />
        <SandboxSection className="bg-km-bg-soft" />
        <SkillsSection />
        <KnowledgeSection className="bg-km-bg-soft" />
        <AlphaEngineSection />
        <KizoNLPSection className="bg-km-bg-soft" />
        <ReportForgeSection />
        <SparkSection className="bg-km-bg-soft" />
        <CaseStudySection />
        <CommunitySection className="bg-km-bg-soft" />
      </main>
      <Footer />
    </div>
  );
}
