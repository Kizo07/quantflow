import { cn } from "@/lib/utils";

export function Section({
  className,
  title,
  subtitle,
  children,
  id,
  kicker,
}: {
  className?: string;
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  children: React.ReactNode;
  id?: string;
  kicker?: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className={cn(
        "mx-auto flex w-full min-w-0 scroll-mt-20 flex-col py-16",
        className,
      )}
    >
      <header className="flex flex-col items-center justify-between px-4">
        {kicker && (
          <p className="text-km-cyan mb-3 font-mono text-xs font-medium tracking-[0.2em] uppercase">
            {kicker}
          </p>
        )}
        <div className="text-foreground mb-4 max-w-full text-center text-3xl font-bold break-words sm:text-4xl md:text-5xl">
          {title}
        </div>
        {subtitle && (
          <div className="text-muted-foreground max-w-full text-center text-base break-words sm:text-lg md:text-xl">
            {subtitle}
          </div>
        )}
      </header>
      <main className="mt-4 w-full min-w-0 px-4">{children}</main>
    </section>
  );
}
