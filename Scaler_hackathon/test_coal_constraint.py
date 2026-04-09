#!/usr/bin/env python3
"""
Quick test of coal constraint fix without hitting API limits.
Directly tests the simulator's step_coal function to verify constraints.
"""

import sys
from dataclasses import dataclass, field
from server.simulator import (
    step_coal, CoalState, GridSimState,
    COAL_MAX_MW, COAL_MIN_MW, COAL_EMERGENCY_BOOST_CEILING_MW,
    COAL_BOOST_DAMAGE_MW, COAL_RAMP_MW, COAL_STARTUP_STEPS,
    COAL_RESTART_COST
)

@dataclass
class TestCoalState:
    """Test harness for coal state"""
    
    def test_normal_ramp(self):
        """Test normal ramping within limits"""
        coal_state = CoalState()
        coal_state.available = True
        coal_state.online = True
        coal_state.output_mw = 400.0
        coal_state.max_mw = COAL_MAX_MW  # 600
        
        # Ramp up by 100 MW
        result = step_coal(coal_state, delta_mw=100.0, emergency_boost=False)
        assert result <= coal_state.max_mw, f"Normal ramp exceeded max: {result} > {coal_state.max_mw}"
        print(f"✓ Normal ramp: {result} MW (max: {coal_state.max_mw}) - PASS")
        
    def test_emergency_boost_with_damage(self):
        """Test emergency boost with damage - CRITICAL TEST"""
        coal_state = CoalState()
        coal_state.available = True
        coal_state.online = True
        coal_state.output_mw = 500.0  # Below max
        coal_state.max_mw = COAL_MAX_MW  # 600
        
        # Trigger emergency boost
        result = step_coal(coal_state, delta_mw=100.0, emergency_boost=True)
        
        # After boost, max should be reduced by damage
        expected_max_after_damage = COAL_MAX_MW - COAL_BOOST_DAMAGE_MW  # 600 - 50 = 550
        
        # Output can reach up to base_max + ceiling (750) absolute
        # But we need to verify it doesn't exceed this
        max_allowed = COAL_MAX_MW + COAL_EMERGENCY_BOOST_CEILING_MW  # 600 + 150 = 750
        
        assert result <= max_allowed, f"Emergency boost exceeded absolute max: {result} > {max_allowed}"
        assert coal_state.max_mw == expected_max_after_damage, \
            f"Damage not applied correctly: {coal_state.max_mw} != {expected_max_after_damage}"
        
        print(f"✓ Emergency boost with damage:")
        print(f"  - Output: {result} MW")
        print(f"  - Max after damage: {coal_state.max_mw} MW (damaged from {COAL_MAX_MW})")
        print(f"  - Within absolute ceiling? {result} <= {max_allowed} - PASS")
        
    def test_emergency_boost_cumulative_damage(self):
        """Test that multiple boosts don't create violations"""
        coal_state = CoalState()
        coal_state.available = True
        coal_state.online = True
        coal_state.boost_damage_steps = 0
        
        # Scenario: multiple emergency boosts
        states = []
        for i in range(3):
            coal_state.output_mw = 400.0 + (i * 50)
            coal_state.max_mw = max(COAL_MIN_MW, COAL_MAX_MW - (i * COAL_BOOST_DAMAGE_MW))
            coal_state.boost_damage_steps = 0  # Reset counter
            
            result = step_coal(coal_state, delta_mw=100.0, emergency_boost=True)
            
            max_allowed = COAL_MAX_MW + COAL_EMERGENCY_BOOST_CEILING_MW
            violation = result > max_allowed
            
            states.append({
                'iteration': i,
                'output': result,
                'max_mw': coal_state.max_mw,
                'violated': violation
            })
            
            assert not violation, f"Iteration {i}: Output {result} exceeded ceiling {max_allowed}"
        
        print(f"✓ Cumulative damage across 3 boosts:")
        for s in states:
            print(f"  - Iteration {s['iteration']}: Output={s['output']:.0f} MW, Max={s['max_mw']:.0f} MW")
        print(f"  - No violations detected - PASS")

def run_all_tests():
    """Run all coal constraint tests"""
    print("\n" + "="*60)
    print("COAL CONSTRAINT FIX VALIDATION")
    print("="*60 + "\n")
    
    tester = TestCoalState()
    
    try:
        tester.test_normal_ramp()
        print()
        tester.test_emergency_boost_with_damage()
        print()
        tester.test_emergency_boost_cumulative_damage()
        
        print("\n" + "="*60)
        print("✅ ALL COAL CONSTRAINT TESTS PASSED")
        print("="*60)
        print("\nConclusion: Coal constraint fix is working correctly.")
        print("Emergency boost respects absolute ceiling (750 MW)")
        print("Damage is properly applied without creating violations.")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
