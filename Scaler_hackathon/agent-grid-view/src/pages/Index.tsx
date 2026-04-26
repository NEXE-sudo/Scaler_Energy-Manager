import { useEffect, useMemo, useState, useCallback } from "react";
import { mockTasks, type Difficulty, type SimulationStep } from "@/lib/simulation-mock";
import { SimulationStepBlock } from "@/components/simulation/SimulationStepBlock";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ChevronLeft,
  ChevronRight,
  Zap,
  Activity,
  Server,
  WifiOff,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Type for history records from base_scores.json
interface HistoryRecord {
  step: number;
  demand: number;
  supply: number;
  frequency: number;
  blackoutRisk?: string;
  planning?: { thought?: string; action?: string };
  dispatch?: { thought?: string; controls?: Record<string, unknown> };
  market?: { thought?: string; controls?: Record<string, unknown> };
  finalAction?: Record<string, unknown>;
}

const difficultyStyles: Record<Difficulty, string> = {
  easy: "bg-stable/15 text-stable border-stable/30",
  medium: "bg-warning/15 text-warning border-warning/30",
  hard: "bg-critical/20 text-critical border-critical/40",
};

// Determine the API base URL  same origin when deployed on HF Spaces,
// localhost fallback for local dev.
const API_BASE =
  import.meta.env.VITE_API_URL ||
  (typeof window !== "undefined" && window.location.hostname !== "localhost"
    ? ""  // same origin on HF Spaces
    : "http://localhost:7860");

type ApiStatus = "checking" | "online" | "offline";

async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    return res.ok;
  } catch {
    return false;
  }
}

async function fetchLiveSteps(taskId: string): Promise<SimulationStep[] | null> {
  try {
    // Reset the environment
    const resetRes = await fetch(`${API_BASE}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId }),
      signal: AbortSignal.timeout(8000),
    });
    if (!resetRes.ok) return null;

    // Fetch baseline results for this task (pre-computed from base_scores.json)
    // We use them to populate the step history display
    return null; // live step streaming is a future enhancement
  } catch {
    return null;
  }
}

// Map base_scores.json history format  SimulationStep[]
function mapHistoryToSteps(history: HistoryRecord[]): SimulationStep[] {
  return history.map((h) => ({
    step: h.step,
    demand: Math.round(h.demand),
    supply: Math.round(h.supply),
    frequency: h.frequency,
    blackoutRisk:
      h.blackoutRisk === "critical"
        ? "critical"
        : h.blackoutRisk === "high"
        ? "high"
        : h.blackoutRisk === "medium" || h.blackoutRisk === "none" && h.frequency < 49.5
        ? "medium"
        : "low",
    planning: {
      thought: h.planning?.thought || "Continuing baseline strategy.",
      action: h.planning?.action || "none",
    },
    dispatch: {
      thought: h.dispatch?.thought || "Optimizing real-time dispatch.",
      controls: h.dispatch?.controls || {},
    },
    market: {
      thought: h.market?.thought || "Managing economic efficiency.",
      controls: h.market?.controls || {},
    },
    finalAction: h.finalAction || {},
    reward: Math.max(0, Math.min(1, (h.reward + 20) / 25)), // normalise raw reward for display
    status:
      h.status === "failure" || h.reward <= -500
        ? "failure"
        : h.frequency < 49.5 || h.blackoutRisk === "critical"
        ? "warning"
        : "stable",
  }));
}

const Index = () => {
  const [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [stepIndexByDifficulty, setStepIndexByDifficulty] = useState<
    Record<Difficulty, number>
  >({ easy: 0, medium: 0, hard: 0 });

  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [liveStepsByTask, setLiveStepsByTask] = useState<
    Partial<Record<Difficulty, SimulationStep[]>>
  >({});
  const [isLoadingLive, setIsLoadingLive] = useState(false);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth().then((ok) => {
      setApiStatus(ok ? "online" : "offline");
    });
  }, []);

  // Try to load live results from the API's pre-computed baseline scores
  const loadLiveResults = useCallback(async () => {
    if (apiStatus !== "online") return;
    setIsLoadingLive(true);
    try {
      const res = await fetch(`${API_BASE}/baseline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tasks: [difficulty] }),
        signal: AbortSignal.timeout(60000),
      });
      if (res.ok) {
        const data = await res.json();
        const results = data.baseline_scores?.results || data.results || {};
        const taskResult = results[difficulty];
        if (taskResult?.history?.length) {
          const steps = mapHistoryToSteps(taskResult.history);
          setLiveStepsByTask((prev) => ({ ...prev, [difficulty]: steps }));
        }
      }
    } catch {
      // silently ignore  fall back to mock
    } finally {
      setIsLoadingLive(false);
    }
  }, [apiStatus, difficulty]);

  // Active steps: live data if available, otherwise mock
  const activeSteps: SimulationStep[] = useMemo(() => {
    return (
      liveStepsByTask[difficulty] ||
      mockTasks.find((t) => t.difficulty === difficulty)!.steps
    );
  }, [difficulty, liveStepsByTask]);

  const isUsingLive = Boolean(liveStepsByTask[difficulty]);

  const task = useMemo(
    () => mockTasks.find((t) => t.difficulty === difficulty)!,
    [difficulty],
  );

  const stepIndex = stepIndexByDifficulty[difficulty];
  const currentStep = activeSteps[stepIndex];

  const goPrev = useCallback(() => {
    setStepIndexByDifficulty((s) => ({
      ...s,
      [difficulty]: Math.max(0, s[difficulty] - 1),
    }));
  }, [difficulty]);

  const goNext = useCallback(() => {
    setStepIndexByDifficulty((s) => ({
      ...s,
      [difficulty]: Math.min(activeSteps.length - 1, s[difficulty] + 1),
    }));
  }, [difficulty, activeSteps.length]);

  // Reset step index when switching difficulty
  useEffect(() => {
    // don't reset  keep per-difficulty state
  }, [difficulty]);

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
  }, [difficulty, activeSteps.length, goPrev, goNext]);

  const canPrev = stepIndex > 0;
  const canNext = stepIndex < activeSteps.length - 1;

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-6 py-12 sm:py-16">
        {/* Header */}
        <header className="mb-10 border-b border-border/60 pb-8">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-stable" />
              <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                Simulation  Live
              </span>
            </div>

            {/* API Status badge */}
            <div className="flex items-center gap-3">
              {apiStatus === "checking" && (
                <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                  <RefreshCw className="h-3 w-3 animate-spin" />
                  Checking API
                </span>
              )}
              {apiStatus === "online" && (
                <span className="flex items-center gap-1.5 rounded-full border border-stable/30 bg-stable/10 px-2.5 py-1 text-[10px] font-medium text-stable">
                  <Server className="h-3 w-3" />
                  Backend Online
                </span>
              )}
              {apiStatus === "offline" && (
                <span className="flex items-center gap-1.5 rounded-full border border-border/60 px-2.5 py-1 text-[10px] text-muted-foreground">
                  <WifiOff className="h-3 w-3" />
                  Using Demo Data
                </span>
              )}

              {apiStatus === "online" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={loadLiveResults}
                  disabled={isLoadingLive}
                  className="h-7 gap-1.5 text-xs"
                >
                  {isLoadingLive ? (
                    <RefreshCw className="h-3 w-3 animate-spin" />
                  ) : (
                    <Zap className="h-3 w-3" />
                  )}
                  {isLoadingLive ? "Running" : "Run Live Baseline"}
                </Button>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary">
              <Activity className="h-5 w-5 text-muted-foreground" />
            </div>
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
                Multi-Agent Grid Simulation
              </h1>
              <p className="mt-0.5 text-sm text-muted-foreground">
                Step-by-step negotiation and decision process {" "}
                {isUsingLive ? (
                  <span className="text-stable">Live API results</span>
                ) : (
                  <span>Demo scenario data</span>
                )}
              </p>
            </div>
          </div>
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
                {liveStepsByTask[t.difficulty as Difficulty] && (
                  <span className="ml-1.5 h-1.5 w-1.5 rounded-full bg-stable" />
                )}
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
                      difficultyStyles[t.difficulty as Difficulty],
                    )}
                  >
                    {t.difficulty}
                  </Badge>
                  <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
                    Task
                  </span>
                  {isUsingLive && t.difficulty === difficulty && (
                    <span className="ml-auto flex items-center gap-1 text-[10px] text-stable">
                      <Zap className="h-3 w-3" />
                      Live results loaded
                    </span>
                  )}
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
                    Step {stepIndex + 1} / {activeSteps.length}
                  </span>
                  <div className="flex items-center gap-1.5">
                    {activeSteps.map((_, i) => (
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
          Use   arrow keys to navigate steps {" "}
          {apiStatus === "online"
            ? "Backend connected  click 'Run Live Baseline' to fetch real results"
            : "Demo data  deploy with API keys for live simulation"}
        </footer>
      </div>
    </div>
  );
};

export default Index;