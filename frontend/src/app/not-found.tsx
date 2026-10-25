import Link from "next/link";

import { Footer } from "@/components/landing/footer";
import { Header } from "@/components/landing/header";
import { Button } from "@/components/ui/button";
import { DEFAULT_LOCALE } from "@/core/i18n/locale";

import { NotFoundBackButton } from "./not-found-back-button";

export default function NotFound() {
  return (
    <div className="bg-km-bg min-h-screen w-full overflow-x-clip">
      <Header locale={DEFAULT_LOCALE} />
      <main className="container-md mx-auto flex min-h-[60vh] flex-col items-center justify-center gap-4 px-4 pt-24 text-center">
        <p className="text-sm font-medium tracking-widest uppercase">404</p>
        <h1 className="font-serif text-3xl">This page could not be found</h1>
        <p className="text-secondary-foreground max-w-md text-sm">
          The page you are looking for does not exist or was moved.
        </p>
        <div className="mt-2 flex items-center gap-3">
          <Button variant="outline" size="sm" asChild>
            <Link href="/">Go home</Link>
          </Button>
          <NotFoundBackButton />
        </div>
      </main>
      <Footer />
    </div>
  );
}
