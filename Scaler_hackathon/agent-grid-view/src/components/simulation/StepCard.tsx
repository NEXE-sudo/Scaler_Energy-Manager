import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { BlackoutRisk, SimulationStep } from "@/lib/simulation-mock";
import { Activity, Gauge, TrendingDown, TrendingUp } from "lucide-react";

const riskStyles: Record<BlackoutRisk, string> = {
  low: "bg-stable/15 text-stable border-stable/30",
  medium: "bg-warning/15 text-warning border-warning/30",
  high: "bg-dispatch/15 text-dispatch border-dispatch/40",
  critical: "bg-critical/20 text-critical border-critical/40",
};

interface Props {
  step: SimulationStep;
}

const Metric = ({
  label,
  value,
  unit,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  unit: string;
  icon: React.ComponentType<{ className?: string }>;
}) => (
  <div className="flex flex-col gap-1.5 rounded-lg border border-border/60 bg-secondary/40 p-4">
    <div className="flex items-center gap-1.5 text-xs uppercase tracking-wider text-muted-foreground">
      <Icon className="h-3.5 w-3.5" />
      {label}
    </div>
    <div className="flex items-baseline gap-1.5">
      <span className="font-mono text-2xl font-semibold tracking-tight text-foreground">
        {value}
      </span>
      <span className="text-xs text-muted-foreground">{unit}</span>
    </div>
  </div>
);

export const StepCard = ({ step }: Props) => {
  const balance = step.supply - step.demand;
  const BalanceIcon = balance >= 0 ? TrendingUp : TrendingDown;

  return (
    <Card className="border-border/60 bg-card p-6 shadow-card">
      <div className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-secondary font-mono text-sm text-muted-foreground">
            #{step.step.toString().padStart(2, "0")}
          </div>
          <div>
            <h3 className="text-sm font-medium text-foreground">Simulation Step</h3>
            <p className="text-xs text-muted-foreground">Grid state snapshot</p>
          </div>
        </div>
        <Badge
          variant="outline"
          className={cn("font-mono text-[10px] uppercase tracking-widest", riskStyles[step.blackoutRisk])}
        >
          {step.blackoutRisk} risk
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Demand" value={step.demand} unit="MW" icon={TrendingDown} />
        <Metric label="Supply" value={step.supply} unit="MW" icon={TrendingUp} />
        <Metric label="Frequency" value={step.frequency.toFixed(2)} unit="Hz" icon={Gauge} />
        <Metric
          label="Balance"
          value={`${balance >= 0 ? "+" : ""}${balance}`}
          unit="MW"
          icon={BalanceIcon}
        />
      </div>

      <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
        <Activity className="h-3.5 w-3.5" />
        Live grid telemetry  updated each simulation tick
      </div>
    </Card>
  );
};
