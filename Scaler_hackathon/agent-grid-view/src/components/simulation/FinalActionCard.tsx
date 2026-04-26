import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { CheckCircle2 } from "lucide-react";

const HIGHLIGHT_KEYS = new Set(["coal_delta", "battery_mode", "demand_response"]);

interface Props {
  action: Record<string, string | number>;
}

export const FinalActionCard = ({ action }: Props) => {
  return (
    <Card className="border-border/60 bg-card p-6 shadow-card">
      <div className="mb-4 flex items-center gap-2">
        <CheckCircle2 className="h-4 w-4 text-foreground/70" />
        <h3 className="text-sm font-semibold uppercase tracking-widest text-foreground">
          Final Decision
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {Object.entries(action).map(([k, v]) => {
          const highlighted = HIGHLIGHT_KEYS.has(k);
          return (
            <div
              key={k}
              className={cn(
                "rounded-md border p-3",
                highlighted
                  ? "border-foreground/20 bg-secondary"
                  : "border-border/60 bg-secondary/30",
              )}
            >
              <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                {k}
              </div>
              <div
                className={cn(
                  "font-mono text-sm",
                  highlighted ? "font-semibold text-foreground" : "text-foreground/80",
                )}
              >
                {v}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
};
