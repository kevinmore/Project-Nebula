/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>

HK_COMPILE_TIME_ASSERT( hkpResponseModifier::MASS_SCALING == 1 );
HK_COMPILE_TIME_ASSERT( hkpResponseModifier::CENTER_OF_MASS_DISPLACEMENT == 2);
HK_COMPILE_TIME_ASSERT( hkpResponseModifier::SURFACE_VELOCITY == 4 );

const hkUint16 hkpResponseModifier::tableOfAdditionalSchemaSizes[] = 
{
	0,
	hkpMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE,
	+ hkpCenterOfMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE,
	+ hkpCenterOfMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE + hkpMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE,
	hkpMovingSurfaceModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE,
	hkpMovingSurfaceModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE + hkpMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE,
	hkpMovingSurfaceModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE + hkpCenterOfMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE,
	hkpMovingSurfaceModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE + hkpCenterOfMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE + hkpMassChangerModifierConstraintAtom::ADDITIONAL_SCHEMA_SIZE,
};

void HK_CALL hkpResponseModifier::setInvMassScalingForContact( hkpDynamicsContactMgr* manager, hkpRigidBody* bodyA, hkpRigidBody* bodyB, hkpConstraintOwner& constraintOwner, const hkVector4& factorA, const hkVector4& factorB )
{

	HK_ASSERT2(0x1a6ebbda, !(bodyA->isFixedOrKeyframed() && factorB.lengthSquared<4>().getReal() == 0.f), "Attempting to collide an object of infinite mass and inertia with a fixed object.");
	HK_ASSERT2(0x21d19edd, !(bodyB->isFixedOrKeyframed() && factorA.lengthSquared<4>().getReal() == 0.f), "Attempting to collide an object of infinite mass and inertia with a fixed object.");

	hkpConstraintInstance* instance = manager->getConstraintInstance();
	if ( !instance )
	{
		return;
	}

	// If the next line fires an assert, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("SetMassChang", HK_NULL);

	//
	// Search existing modifier list for matching type
	//
	hkpModifierConstraintAtom* modifier = hkpWorldConstraintUtil::findModifier( instance, hkpConstraintAtom::TYPE_MODIFIER_MASS_CHANGER );
	hkpMassChangerModifierConstraintAtom* massChangermodifier = reinterpret_cast<hkpMassChangerModifierConstraintAtom*>(modifier);

	if ( !massChangermodifier )
	{
		//
		// Build and insert new modifier atom
		//
		massChangermodifier = new hkpMassChangerModifierConstraintAtom;
		hkpWorldConstraintUtil::addModifier( instance, constraintOwner, massChangermodifier );
	}

	// Set modifier data
	if (bodyA == instance->getEntityA())
	{
		massChangermodifier->m_factorA = factorA;
		massChangermodifier->m_factorB = factorB;
	}
	else
	{
		massChangermodifier->m_factorA = factorB;
		massChangermodifier->m_factorB = factorA;
	}

	HK_TIMER_END();
}


void HK_CALL hkpResponseModifier::setInvMassScalingForContact( hkpDynamicsContactMgr* manager, hkpRigidBody* body, hkpConstraintOwner& constraintOwner, const hkVector4& factor )
{
	hkpConstraintInstance* instance = manager->getConstraintInstance();
	if ( !instance )
	{
		return;
	}

	// If the next line fires an assert, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("SetMassChang", HK_NULL);

	//
	// Search existing modifier list for matching type
	//
	hkpModifierConstraintAtom* modifier = hkpWorldConstraintUtil::findModifier( instance, hkpConstraintAtom::TYPE_MODIFIER_MASS_CHANGER );
	hkpMassChangerModifierConstraintAtom* massChangermodifier = reinterpret_cast<hkpMassChangerModifierConstraintAtom*>(modifier);

	const hkVector4 one = hkVector4::getConstant<HK_QUADREAL_1>();
	if ( !massChangermodifier )
	{
		//
		// Build and insert new modifier atom
		//
		massChangermodifier = new hkpMassChangerModifierConstraintAtom;
		hkpWorldConstraintUtil::addModifier( instance, constraintOwner, massChangermodifier );
		if ( body == instance->getEntityA() )
		{
			massChangermodifier->m_factorA = factor;
			massChangermodifier->m_factorB = one;
		}
		else
		{
			massChangermodifier->m_factorA = one;
			massChangermodifier->m_factorB = factor;
		}
	}
	else
	{
		if ( body == instance->getEntityA() )
		{
			massChangermodifier->m_factorA = factor;
		}
		else
		{
			massChangermodifier->m_factorB = factor;
		}
		
		// Check that both entities have not been scaled to have infinite mass.
		if ( massChangermodifier->m_factorA.getW().isEqualZero() && massChangermodifier->m_factorB.getW().isEqualZero() )
		{
			// Rescale both components back to 1.0f.
			massChangermodifier->m_factorA.setW(one);
			massChangermodifier->m_factorB.setW(one);
		}
	}

	HK_TIMER_END();
}


void HK_CALL hkpResponseModifier::setCenterOfMassDisplacementForContact( hkpDynamicsContactMgr* manager, hkpRigidBody* bodyA, hkpRigidBody* bodyB, hkpConstraintOwner& constraintOwner, const hkVector4& displacementA, const hkVector4& displacementB )
{
	hkpConstraintInstance* instance = manager->getConstraintInstance();
	if ( !instance )
	{
		return;
	}

	// If the next line fires an assert, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("SetMassChang", HK_NULL);

	//
	// Search existing modifier list for matching type
	//
	hkpModifierConstraintAtom* modifier = hkpWorldConstraintUtil::findModifier( instance, hkpConstraintAtom::TYPE_MODIFIER_CENTER_OF_MASS_CHANGER );
	hkpCenterOfMassChangerModifierConstraintAtom* centerOfMassModifier = reinterpret_cast<hkpCenterOfMassChangerModifierConstraintAtom*>(modifier);

	if ( !centerOfMassModifier )
	{
		//
		// Build and insert new modifier atom
		//
		centerOfMassModifier = new hkpCenterOfMassChangerModifierConstraintAtom;
		hkpWorldConstraintUtil::addModifier( instance, constraintOwner, centerOfMassModifier );
	}

	// Set modifier data
	if (bodyA == instance->getEntityA())
	{
		centerOfMassModifier->m_displacementA = displacementA;
		centerOfMassModifier->m_displacementB = displacementB;
	}
	else
	{
		centerOfMassModifier->m_displacementA = displacementB;
		centerOfMassModifier->m_displacementB = displacementA;
	}

	HK_TIMER_END();
}


void HK_CALL hkpResponseModifier::setImpulseScalingForContact( hkpDynamicsContactMgr* manager, hkpRigidBody* bodyA, hkpRigidBody* bodyB, hkpConstraintOwner& constraintOwner, hkReal usedImpulseFraction, hkReal maxAcceleration )
{
	hkpConstraintInstance* instance = manager->getConstraintInstance();
	if ( !instance )
	{
		return;
	}

		// If the next line fires an assert, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("SetSoftContact", HK_NULL);

	//
	// Search existing modifier list for matching type (and update, if already present)
	//
	{
		hkpModifierConstraintAtom* container = hkpWorldConstraintUtil::findModifier( instance, hkpConstraintAtom::TYPE_MODIFIER_SOFT_CONTACT );
		if ( container )
		{
			
			hkpSoftContactModifierConstraintAtom* softContactContainer = reinterpret_cast<hkpSoftContactModifierConstraintAtom*>(container);
			softContactContainer->m_tau = usedImpulseFraction;
			softContactContainer->m_maxAcceleration = maxAcceleration;

			goto END;
		}
	}

	//
	// Build and insert new modifier atom
	//
	{
		hkpSoftContactModifierConstraintAtom* softContactContainer = new hkpSoftContactModifierConstraintAtom;
		softContactContainer->m_tau = usedImpulseFraction;
		softContactContainer->m_maxAcceleration = maxAcceleration;
		hkpWorldConstraintUtil::addModifier( instance, constraintOwner, softContactContainer );
	}

END:
	HK_TIMER_END();
}



void HK_CALL hkpResponseModifier::setSurfaceVelocity( hkpDynamicsContactMgr* manager, hkpRigidBody* body, hkpConstraintOwner& constraintOwner, const hkVector4& velWorld )
{
	hkpConstraintInstance* instance = manager->getConstraintInstance();
	if ( !instance )
	{
		return;
	}

		// If any of these asserts gets fired, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("SetSurfVel", HK_NULL);

	hkVector4 velocity = velWorld;
	if (instance->getEntityA() == body )
	{
		velocity.setNeg<4>(velocity);
	}


	//
	// Search existing modifier list for matching type (and update, if already present)
	//
	{
		hkpModifierConstraintAtom* modifier = hkpWorldConstraintUtil::findModifier( instance, hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE );
		if ( modifier )
		{
			hkpMovingSurfaceModifierConstraintAtom* movingSurfaceContainer = reinterpret_cast<hkpMovingSurfaceModifierConstraintAtom*>(modifier);
			movingSurfaceContainer->getVelocity() = velocity;
			goto END;
		}
	}

	//
	// Build and insert new modifier atom
	//
	{
		hkpMovingSurfaceModifierConstraintAtom* movingSurfaceContainer = new hkpMovingSurfaceModifierConstraintAtom;
		movingSurfaceContainer->getVelocity() = velocity;
		hkpWorldConstraintUtil::addModifier( instance, constraintOwner, movingSurfaceContainer );
	}

END:

	HK_TIMER_END();
}



void HK_CALL hkpResponseModifier::clearSurfaceVelocity( hkpDynamicsContactMgr* manager, hkpConstraintOwner& constraintOwner, hkpRigidBody* body )
{
	hkpConstraintInstance* instance = manager->getConstraintInstance();
	if ( !instance )
	{
		return;
	}

	// If you run onto those asserts, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("ClrSurfVel", HK_NULL);

		// destroy modifier atom
	hkpWorldConstraintUtil::removeAndDeleteModifier( instance, constraintOwner, hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE );

	HK_TIMER_END();
}

void HK_CALL hkpResponseModifier::setLowSurfaceViscosity( hkpDynamicsContactMgr* manager, hkpConstraintOwner& constraintOwner )
{
	hkpConstraintInstance* instance = manager->getConstraintInstance();
	if ( !instance )
	{
		return;
	}

		// If you get those asserts, check the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("SetSurfVisc", HK_NULL);

	//
	// Search existing modifier list for matching type
	//
	{
		hkpModifierConstraintAtom* modifier = hkpWorldConstraintUtil::findModifier( instance, hkpConstraintAtom::TYPE_MODIFIER_VISCOUS_SURFACE );
		if ( modifier )
		{
			goto END;
		}
	}

	//
	// Build and insert new modifier atom
	//
	{
		hkpViscousSurfaceModifierConstraintAtom* viscousSurfaceContainer = new hkpViscousSurfaceModifierConstraintAtom;
		hkpWorldConstraintUtil::addModifier( instance, constraintOwner, viscousSurfaceContainer );
	}

END:

	HK_TIMER_END();
}

void HK_CALL hkpResponseModifier::disableConstraint(hkpConstraintInstance* instance, hkpConstraintOwner& constraintOwner)
{
		// If any of these asserts gets fired, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("DsblConstr", HK_NULL);

	// Search existing modifier list for matching type
	//
	hkpModifierConstraintAtom* modifier = hkpWorldConstraintUtil::findModifier( instance, hkpConstraintAtom::TYPE_MODIFIER_IGNORE_CONSTRAINT );
	if ( !modifier )
	{
		// Build and insert new modifier atom
		//
		hkpIgnoreModifierConstraintAtom* ignoreModifier = new hkpIgnoreModifierConstraintAtom;
		hkpWorldConstraintUtil::addModifier( instance, constraintOwner, ignoreModifier );
	}
	
	HK_TIMER_END();
}

void HK_CALL hkpResponseModifier::enableConstraint(hkpConstraintInstance* instance, hkpConstraintOwner& constraintOwner)
{
		// If you run onto those asserts, read the hkpResponseModifier reference manual
	constraintOwner.checkAccessRw();

	HK_TIMER_BEGIN("EnblConstr", HK_NULL);

	// destroy modifier atom
	hkpWorldConstraintUtil::removeAndDeleteModifier( instance, constraintOwner, hkpConstraintAtom::TYPE_MODIFIER_IGNORE_CONSTRAINT );

	// Zero solver results to prevent friction or motors from snapping on the first frame after activation.
	hkpSolverResults* results = reinterpret_cast<hkpSolverResults*>(instance->getRuntime());
	if (results)
	{
		hkpConstraintData::RuntimeInfo info;
		hkBool wantRuntime = true;
		instance->getData()->getRuntimeInfo(wantRuntime, info);

		for (int i = 0; i < info.m_numSolverResults; i++)
		{
			results[i].init();
		}
	}

	HK_TIMER_END();
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
