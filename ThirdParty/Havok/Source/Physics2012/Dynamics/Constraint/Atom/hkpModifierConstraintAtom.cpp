/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpModifierConstraintAtom.h>

#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>

#include <Common/Base/Math/Matrix/hkMatrix3Util.h>


void hkpMassChangerModifierConstraintAtom::collisionResponseBeginCallback( const hkContactPoint& cp, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB )
{
	hkVector4 factor = m_factorA;
	hkpSimpleConstraintInfoInitInput* in = &inA;
	for (int i = 0; i < 2; i++)
	{
		in->m_invMasses.mul( factor );

		if (factor.allComponentsEqual<3>())
		{
			in->m_invInertia.mul( factor.getComponent<0>() );
		}
		else
		{
			//
			// Do non-uniform inertia scaling
			//

			// Get local diagonal inertia tensor from world
			const hkRotation& rotation = in->m_transform->getRotation();

			hkMatrix3 tmp; 
			tmp.setMulInverseMul(rotation, in->m_invInertia);
			hkMatrix3 localInvMassesMtx;
			localInvMassesMtx.setMul(tmp, rotation);

			// Scale
			hkVector4 diag; hkMatrix3Util::_getDiagonal(localInvMassesMtx, diag); diag.mul(factor);
			hkMatrix3Util::_setDiagonalOnly(diag, localInvMassesMtx);

			// Get inertia in world again.
			tmp.setMulInverse(localInvMassesMtx, rotation);
			in->m_invInertia.setMul(rotation, tmp);
		}

		// 2nd body
		factor = m_factorB;
		in = &inB;
	}
}

void hkpMassChangerModifierConstraintAtom::collisionResponseEndCallback( const hkContactPoint& cp, hkReal impulseApplied, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB )
{
}

void hkpCenterOfMassChangerModifierConstraintAtom::collisionResponseBeginCallback( const hkContactPoint& cp, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB )
{
	hkVector4 displacementA; displacementA._setRotatedDir(inA.m_transform->getRotation(), m_displacementA);
	hkVector4 displacementB; displacementB._setRotatedDir(inB.m_transform->getRotation(), m_displacementB);
	inA.m_massRelPos.sub(displacementA);
	inB.m_massRelPos.sub(displacementB);

	// Modify velocity
	hkVector4 velADiff; velADiff.setCross(displacementA, velA.m_angular);
	hkVector4 velBDiff; velBDiff.setCross(displacementB, velB.m_angular);
	velA.m_linear.sub(velADiff);
	velB.m_linear.sub(velBDiff);
}

void hkpCenterOfMassChangerModifierConstraintAtom::collisionResponseEndCallback( const hkContactPoint& cp, hkReal impulseApplied, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB )
{
	hkVector4 displacementA; displacementA._setRotatedDir(inA.m_transform->getRotation(), m_displacementA);
	hkVector4 displacementB; displacementB._setRotatedDir(inB.m_transform->getRotation(), m_displacementB);
	hkVector4 velADiff; velADiff.setCross(displacementA, velA.m_angular);
	hkVector4 velBDiff; velBDiff.setCross(displacementB, velB.m_angular);
	velA.m_linear.add(velADiff);
	velB.m_linear.add(velBDiff);
}

static hkpBodyVelocity s_bodyVelocities[2];
static hkBool s_bodyVelocitiesInitialized = false;

void hkpSoftContactModifierConstraintAtom::collisionResponseBeginCallback( const hkContactPoint& cp, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB)
{
	HK_WARN_ONCE(0x12015a6e, "Using soft contacts (scaling of response force) in TOI events. This may cause a significant performance drop. This pair of bodies should not use continuous collision detection. Reduce the quality type of either of the bodies.");
	HK_ASSERT2(0x172400f2, s_bodyVelocitiesInitialized == false, "hkSoftContactConstraintData uses static variables for processing of TOI collision. It assumed that TOI are always processed in series one after another. If that's changed this class has to be rewritten. ");
	s_bodyVelocities[0] = velA;
	s_bodyVelocities[1] = velB;
}

void hkpSoftContactModifierConstraintAtom::collisionResponseEndCallback( const hkContactPoint& cp, hkReal impulseApplied, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB )
{
	hkSimdReal usedImpulseFraction; usedImpulseFraction.setFromFloat( this->m_tau );

	hkpBodyVelocity& oldA = s_bodyVelocities[0];
	hkpBodyVelocity& oldB = s_bodyVelocities[1];

	velA.m_linear.setInterpolate(oldA.m_linear, velA.m_linear, usedImpulseFraction);
	velA.m_angular.setInterpolate(oldA.m_angular, velA.m_angular, usedImpulseFraction);
	velB.m_linear.setInterpolate(oldB.m_linear, velB.m_linear, usedImpulseFraction);
	velB.m_angular.setInterpolate(oldB.m_angular, velB.m_angular, usedImpulseFraction);
}

void hkpMovingSurfaceModifierConstraintAtom::collisionResponseBeginCallback( const hkContactPoint& cp, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB)
{
	hkVector4 vel = getVelocity();
	// project velocity into the contact plane (so that objects do not sink in)
	const hkSimdReal dot = vel.dot<3>(cp.getNormal());
	hkVector4 perp; perp.setMul( dot, vel );
	vel.sub( perp );
	velB.m_linear.add( vel );
}

void hkpMovingSurfaceModifierConstraintAtom::collisionResponseEndCallback( const hkContactPoint& cp, hkReal impulseApplied, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB )
{
	hkVector4 vel = getVelocity();

	// project velocity into the contact plane (so that objects do not sink in)
	const hkSimdReal dot = vel.dot<3>(cp.getNormal());
	hkVector4 perp; perp.setMul( dot, vel );
	vel.sub( perp );
	velB.m_linear.sub( vel );
}

int hkpIgnoreModifierConstraintAtom::getConstraintInfo( hkpConstraintInfo& info ) const
{	
	// Get the final atom (non-modifier)
	//
	const hkpConstraintAtom* leaf = this;
	int leafSize = 0;
	while ( leaf->isModifierType() )
	{
		leafSize = reinterpret_cast<const hkpModifierConstraintAtom*>(leaf)->m_childSize;
		leaf = reinterpret_cast<const hkpModifierConstraintAtom*>(leaf)->m_child;
		HK_ASSERT2(0xad3423ab, leaf, "This function can only be called once the modifier is attached to a constraint.");
	}

	HK_ASSERT2(0xad875533, leafSize, "Leaf atoms size is zero.");

	// This will be set to the negative of what's needed by the final atom; unless it's a contact atom. 
	// As contact atoms change size, the we set the size of this modifier to 0.
	//
	// Also, we ignore size of other modifiers as they can be added and removed, and they cause individual calls to add/subConstraintInfo.
	//
	if (leaf->getType() != hkpConstraintAtom::TYPE_CONTACT && leaf->getType() != hkpConstraintAtom::TYPE_BRIDGE)
	{
		hkpConstraintData::ConstraintInfo leafInfo;
		hkpConstraintData::getConstraintInfoUtil(leaf, leafSize, leafInfo);

		// Subtract leaf schema sizes.
		info.m_numSolverElemTemps -= leafInfo.m_numSolverElemTemps;
		info.m_numSolverResults   -= leafInfo.m_numSolverResults;
		info.m_sizeOfSchemas      -= leafInfo.m_sizeOfSchemas;
	}

	return hkpConstraintAtom::CALLBACK_REQUEST_NONE;		
}

int hkpModifierConstraintAtom::addModifierDataToConstraintInfo( hkpConstraintInfo& cinfo, hkUint8& usedFlagsOut ) const
{
	int callBackRequest = CALLBACK_REQUEST_NONE;
	const hkpModifierConstraintAtom* modifier = this;

	switch( modifier->getType() )
	{

	case hkpConstraintAtom::TYPE_MODIFIER_VISCOUS_SURFACE:
		{
			const hkpViscousSurfaceModifierConstraintAtom* m = static_cast<const hkpViscousSurfaceModifierConstraintAtom*>( modifier );
			callBackRequest |= m->getConstraintInfo ( cinfo );
			usedFlagsOut |= hkpResponseModifier::VISCOUS_SURFACE;
			break;
		}
	case hkpConstraintAtom::TYPE_MODIFIER_SOFT_CONTACT:
		{
			const hkpSoftContactModifierConstraintAtom* m = static_cast<const hkpSoftContactModifierConstraintAtom*>( modifier );
			callBackRequest |= m->getConstraintInfo ( cinfo );
			usedFlagsOut |= hkpResponseModifier::IMPULSE_SCALING;
			break;
		}
	case hkpConstraintAtom::TYPE_MODIFIER_MASS_CHANGER:
		{
			const hkpMassChangerModifierConstraintAtom* m = static_cast<const hkpMassChangerModifierConstraintAtom*>( modifier );
			callBackRequest |= m->getConstraintInfo ( cinfo );
			usedFlagsOut |= hkpResponseModifier::MASS_SCALING;
			break;
		}
	case hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE:
		{
			const hkpMovingSurfaceModifierConstraintAtom* m = static_cast<const hkpMovingSurfaceModifierConstraintAtom*>( modifier );
			callBackRequest |= m->getConstraintInfo ( cinfo );
			usedFlagsOut |= hkpResponseModifier::SURFACE_VELOCITY;
			break;
		}
	case hkpConstraintAtom::TYPE_MODIFIER_IGNORE_CONSTRAINT:
		{
			const hkpIgnoreModifierConstraintAtom* m = static_cast<const hkpIgnoreModifierConstraintAtom*>( modifier );
			callBackRequest |= m->getConstraintInfo ( cinfo );
			break;
		}
	case hkpConstraintAtom::TYPE_MODIFIER_CENTER_OF_MASS_CHANGER:
		{
			const hkpCenterOfMassChangerModifierConstraintAtom* m = static_cast<const hkpCenterOfMassChangerModifierConstraintAtom*>( modifier );
			callBackRequest |= m->getConstraintInfo ( cinfo );
			usedFlagsOut |= hkpResponseModifier::CENTER_OF_MASS_DISPLACEMENT;
			break;
		}

	default:
		{
			HK_ASSERT2(0xaf673682, 0, "Unknown constraint modifier type.");
			break;
		}
	}
	return callBackRequest;
}

int HK_CALL hkpModifierConstraintAtom::addAllModifierDataToConstraintInfo( hkpModifierConstraintAtom* modifier, hkpConstraintInfo& cinfo, hkUint8& usedFlagsOut )
{
	int callBackRequest = CALLBACK_REQUEST_NONE;
	
	hkpConstraintAtom* atom = modifier;

	while ( 1 )
	{
		// abort if we reached the constraint's original atom
		if ( !atom->isModifierType() )
		{
			break;
		}
		hkpModifierConstraintAtom* mac = reinterpret_cast<hkpModifierConstraintAtom*>(atom);

		callBackRequest |= mac->addModifierDataToConstraintInfo( cinfo, usedFlagsOut );

		atom = mac->m_child;
	}
	return callBackRequest;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
