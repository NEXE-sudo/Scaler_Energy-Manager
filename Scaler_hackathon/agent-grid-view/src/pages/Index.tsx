import { useEffect, useMemo, useState } from "react";
import { mockTasks, type Difficulty } from "@/lib/simulation-mock";
import { SimulationStepBlock } from "@/components/simulation/SimulationStepBlock";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

const difficultyStyles: Record<Difficulty, string> = {
  easy: "bg-stable/15 text-stable border-stable/30",
  medium: "bg-warning/15 text-warning border-warning/30",
  hard: "bg-critical/20 text-critical border-critical/40",
};

const Index = () => {
  const [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [stepIndexByDifficulty, setStepIndexByDifficulty] = useState<
    Record<Difficulty, number>
  >({ easy: 0, medium: 0, hard: 0 });

  const task = useMemo(
    () => mockTasks.find((t) => t.difficulty === difficulty)!,
    [difficulty],
  );
  const stepIndex = stepIndexByDifficulty[difficulty];
  const currentStep = task.steps[stepIndex];

  const goPrev = () =>
    setStepIndexByDifficulty((s) => ({
      ...s,
      [difficulty]: Math.max(0, s[difficulty] - 1),
    }));
  const goNext = () =>
    setStepIndexByDifficulty((s) => ({
      ...s,
      [difficulty]: Math.min(task.steps.length - 1, s[difficulty] + 1),
    }));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goPrev();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goNext();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [difficulty, task.steps.length]);

  const canPrev = stepIndex > 0;
  const canNext = stepIndex < task.steps.length - 1;

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-6 py-12 sm:py-16">
        {/* Header */}
        <header className="mb-10 border-b border-border/60 pb-8">
          <div className="mb-3 flex items-center gap-2">
            <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-stable" />
            <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              Simulation · Live
            </span>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
            Multi-Agent Grid Simulation
          </h1>
          <p className="mt-2 text-sm text-muted-foreground sm:text-base">
            Step-by-step negotiation and decision process
          </p>
        </header>

        {/* Difficulty tabs */}
        <Tabs
          value={difficulty}
          onValueChange={(v) => setDifficulty(v as Difficulty)}
          className="space-y-6"
        >
          <TabsList className="grid w-full grid-cols-3 sm:w-auto sm:inline-grid">
            {mockTasks.map((t) => (
              <TabsTrigger
                key={t.difficulty}
                value={t.difficulty}
                className="capitalize"
              >
                {t.difficulty}
              </TabsTrigger>
            ))}
          </TabsList>

          {mockTasks.map((t) => (
            <TabsContent
              key={t.difficulty}
              value={t.difficulty}
              className="space-y-6"
            >
              {/* Task description */}
              <Card className="border-border/60 bg-card p-6 shadow-card">
                <div className="mb-3 flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className={cn(
                      "font-mono text-[10px] uppercase tracking-widest",
                      difficultyStyles[t.difficulty],
                    )}
                  >
                    {t.difficulty}
                  </Badge>
                  <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
                    Task
                  </span>
                </div>
                <h2 className="text-lg font-semibold tracking-tight text-foreground">
                  {t.title}
                </h2>
                <p className="mt-1.5 text-sm text-muted-foreground">
                  {t.description}
                </p>
              </Card>

              {/* Step navigation */}
              <div className="flex items-center justify-between gap-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goPrev}
                  disabled={!canPrev}
                  aria-label="Previous step"
                >
                  <ChevronLeft />
                  Previous
                </Button>
                <div className="flex flex-col items-center gap-1.5">
                  <span className="font-mono text-xs text-muted-foreground">
                    Step {stepIndex + 1} / {task.steps.length}
                  </span>
                  <div className="flex items-center gap-1.5">
                    {task.steps.map((_, i) => (
                      <button
                        key={i}
                        onClick={() =>
                          setStepIndexByDifficulty((s) => ({
                            ...s,
                            [difficulty]: i,
                          }))
                        }
                        aria-label={`Go to step ${i + 1}`}
                        className={cn(
                          "h-1.5 rounded-full transition-all",
                          i === stepIndex
                            ? "w-6 bg-foreground"
                            : "w-1.5 bg-border hover:bg-muted-foreground",
                        )}
                      />
                    ))}
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goNext}
                  disabled={!canNext}
                  aria-label="Next step"
                >
                  Next
                  <ChevronRight />
                </Button>
              </div>

              {/* Current step */}
              {currentStep && <SimulationStepBlock step={currentStep} />}
            </TabsContent>
          ))}
        </Tabs>

        <footer className="mt-16 border-t border-border/60 pt-6 text-center text-xs text-muted-foreground">
          Use ← → arrow keys to navigate steps · Mock data
        </footer>
      </div>
    </div>
  );
};

export default Index;
