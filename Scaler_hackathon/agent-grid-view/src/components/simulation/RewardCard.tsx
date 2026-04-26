import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { StepStatus } from "@/lib/simulation-mock";
import { Award } from "lucide-react";

const statusStyles: Record<StepStatus, { badge: string; label: string }> = {
  stable: { badge: "bg-stable/15 text-stable border-stable/30", label: "Stable" },
  warning: { badge: "bg-warning/15 text-warning border-warning/30", label: "Warning" },
  failure: { badge: "bg-critical/20 text-critical border-critical/40", label: "Failure" },
};

interface Props {
  reward: number;
  status: StepStatus;
}

export const RewardCard = ({ reward, status }: Props) => {
  const s = statusStyles[status];
  // Reward bar: clamp 0..1
  const pct = Math.max(0, Math.min(1, reward)) * 100;

  return (
    <Card className="border-border/60 bg-card p-6 shadow-card">
      <div className="flex items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-secondary">
            <Award className="h-4 w-4 text-foreground/70" />
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Reward
            </div>
            <div className="font-mono text-2xl font-semibold text-foreground">
              {reward.toFixed(2)}
            </div>
          </div>
        </div>

        <div className="hidden flex-1 sm:block">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                status === "stable" && "bg-stable",
                status === "warning" && "bg-warning",
                status === "failure" && "bg-critical",
              )}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        <Badge
          variant="outline"
          className={cn("font-mono text-[10px] uppercase tracking-widest", s.badge)}
        >
          {s.label}
        </Badge>
      </div>
    </Card>
  );
};
