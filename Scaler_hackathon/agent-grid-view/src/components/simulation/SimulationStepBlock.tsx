import type { SimulationStep } from "@/lib/simulation-mock";
import { StepCard } from "./StepCard";
import { AgentPanel } from "./AgentPanel";
import { FinalActionCard } from "./FinalActionCard";
import { RewardCard } from "./RewardCard";

interface Props {
  step: SimulationStep;
}

export const SimulationStepBlock = ({ step }: Props) => {
  return (
    <section className="space-y-4">
      <StepCard step={step} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <AgentPanel variant="planning" label="Planning" agent={step.planning} />
        <AgentPanel variant="dispatch" label="Dispatch" agent={step.dispatch} />
        <AgentPanel variant="market" label="Market" agent={step.market} />
      </div>

      <FinalActionCard action={step.finalAction} />
      <RewardCard reward={step.reward} status={step.status} />
    </section>
  );
};
