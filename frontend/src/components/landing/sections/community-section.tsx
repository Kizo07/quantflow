import { GitHubLogoIcon } from "@radix-ui/react-icons";
import Link from "next/link";

import { Button } from "@/components/ui/button";

import { Section } from "../section";

export function CommunitySection({ className }: { className?: string }) {
  return (
    <Section
      id="community"
      className={className}
      kicker="Open source"
      title="Join the Community"
      subtitle="Contribute brilliant ideas to shape the future of QuantFlow. Collaborate, innovate, and make impacts."
    >
      <div className="flex justify-center">
        <Button className="text-xl" size="lg" asChild>
          <Link
            href="https://github.com/Kizo07/quantflow"
            target="_blank"
            rel="noopener noreferrer"
          >
            <GitHubLogoIcon />
            Contribute Now
          </Link>
        </Button>
      </div>
    </Section>
  );
}
