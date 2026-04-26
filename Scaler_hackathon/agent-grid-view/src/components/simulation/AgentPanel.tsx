import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { AgentState } from "@/lib/simulation-mock";

type AgentVariant = "planning" | "dispatch" | "market";

const variantStyles: Record<
  AgentVariant,
  { ring: string; bg: string; dot: string; text: string; border: string }
> = {
  planning: {
    ring: "ring-planning/30",
    bg: "bg-planning-bg",
    dot: "bg-planning",
    text: "text-planning",
    border: "border-planning/20",
  },
  dispatch: {
    ring: "ring-dispatch/30",
    bg: "bg-dispatch-bg",
    dot: "bg-dispatch",
    text: "text-dispatch",
    border: "border-dispatch/20",
  },
  market: {
    ring: "ring-market/30",
    bg: "bg-market-bg",
    dot: "bg-market",
    text: "text-market",
    border: "border-market/20",
  },
};

interface Props {
  variant: AgentVariant;
  label: string;
  agent: AgentState;
}

export const AgentPanel = ({ variant, label, agent }: Props) => {
  const s = variantStyles[variant];

  return (
    <Card className={cn("flex flex-col gap-4 border bg-card p-5 shadow-card", s.border)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={cn("h-2 w-2 rounded-full", s.dot)} />
          <span className={cn("text-xs font-semibold uppercase tracking-widest", s.text)}>
            {label}
          </span>
        </div>
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Agent</span>
      </div>

      <div className="space-y-1.5">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Thought</div>
        <p className="text-sm leading-relaxed text-foreground/90">{agent.thought}</p>
      </div>

      {agent.action && (
        <div
          className={cn(
            "rounded-md px-3 py-2.5 text-sm font-mono",
            s.bg,
            s.text,
          )}
        >
           {agent.action}
        </div>
      )}

      {agent.controls && (
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Key controls
          </div>
          <div className="space-y-1 rounded-md border border-border/60 bg-secondary/40 p-3">
            {Object.entries(agent.controls).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between gap-3 text-xs">
                <span className="font-mono text-muted-foreground">{k}</span>
                <span className={cn("font-mono font-medium", s.text)}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
};
