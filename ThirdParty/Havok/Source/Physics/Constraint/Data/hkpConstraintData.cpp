/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/hkpConstraintData.h>

#if !defined(HK_PLATFORM_SPU)

hkpConstraintData::hkpConstraintData( hkFinishLoadedObjectFlag f )
	: hkReferencedObject(f)
{

}

#endif

void hkpConstraintData::addInstance( hkpConstraintRuntime* runtime, int sizeOfRuntime ) const
{
	if ( runtime )
	{
		hkString::memSet( runtime, 0, sizeOfRuntime );
	}
}

hkpSolverResults* hkpConstraintData::getSolverResults( hkpConstraintRuntime* runtime )
{
	return reinterpret_cast<hkpSolverResults*>(runtime);
}


#define HK_SKIP_ATOM_BY_TYPE(atomType, atomClassName)\
{\
case hkpConstraintAtom::atomType:\
{\
	const atomClassName* atom = static_cast<const atomClassName*>(currentAtom);\
	currentAtom = atom->next();\
}\
	break;\
}

#define HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE(atomType, atomClassName)\
{\
case hkpConstraintAtom::atomType:\
	{\
		const atomClassName* atom = static_cast<const atomClassName*>(currentAtom);\
		atom->addToConstraintInfo(infoOut);\
		currentAtom = atom->next();\
	}\
	break;\
}

void hkpConstraintData::getConstraintInfoUtil( const hkpConstraintAtom* atoms, int sizeOfAllAtoms, hkpConstraintData::ConstraintInfo& infoOut )
{
	infoOut.m_atoms = const_cast<hkpConstraintAtom*>(atoms);
	infoOut.m_sizeOfAllAtoms = sizeOfAllAtoms;
	infoOut.clear();

	// Contact constraints should use hkpSimpleContactConstraintData::getConstraintInfo().
	HK_ASSERT(0x7896e9f7, atoms->m_type != hkpConstraintAtom::TYPE_CONTACT);

	infoOut.addHeader();

	const hkpConstraintAtom* atomsEnd = hkAddByteOffsetConst<const hkpConstraintAtom>( atoms, sizeOfAllAtoms );
	for( const hkpConstraintAtom* currentAtom = atoms; currentAtom < atomsEnd; )
	{
NEXT_SWITCH:
		switch(currentAtom->m_type)
		{
		case hkpConstraintAtom::TYPE_INVALID:
			{
				// If this is blank padding between atoms, then move to the next 16-byte aligned atom
				currentAtom = reinterpret_cast<hkpConstraintAtom*>( HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, hkUlong(currentAtom)) );
				goto NEXT_SWITCH;
			}

		case hkpConstraintAtom::TYPE_BALL_SOCKET:
				{
					const hkpBallSocketConstraintAtom* atom = static_cast<const hkpBallSocketConstraintAtom*>(currentAtom);
					atom->addToConstraintInfo(infoOut);
					const int schemaSize = hkMath::_max2<int>(3 * hkpJacobianSchemaInfo::Bilateral1D::Sizeof, hkpJacobianSchemaInfo::StableBallSocket::Sizeof);
					infoOut.m_extraSchemaSize += hkpJacobianSchemaInfo::NpStableBallSocket::Sizeof - schemaSize;
					currentAtom = atom->next();
				}
				break;

		case hkpConstraintAtom::TYPE_3D_ANG:
			{
				const hkp3dAngConstraintAtom* atom = static_cast<const hkp3dAngConstraintAtom*>(currentAtom);
				atom->addToConstraintInfo(infoOut);
				infoOut.m_extraSchemaSize += hkpJacobianSchemaInfo::NpStableAngular3D::Sizeof - hkpJacobianSchemaInfo::StableAngular3D::Sizeof;
				currentAtom = atom->next();
			}
			break;

		case hkpConstraintAtom::TYPE_DEFORMABLE_3D_LIN:
			{
				const hkpDeformableLinConstraintAtom* atom = static_cast<const hkpDeformableLinConstraintAtom*>(currentAtom);
				atom->addToConstraintInfo(infoOut);
				infoOut.m_extraSchemaSize += hkpJacobianSchemaInfo::NpDeformableLinear3D::Sizeof - hkpJacobianSchemaInfo::DeformableLinear3D::Sizeof;
				currentAtom = atom->next();
			}
			break;

		case hkpConstraintAtom::TYPE_DEFORMABLE_3D_ANG:
			{
				const hkpDeformableAngConstraintAtom* atom = static_cast<const hkpDeformableAngConstraintAtom*>(currentAtom);
				atom->addToConstraintInfo(infoOut);
				infoOut.m_extraSchemaSize += hkpJacobianSchemaInfo::NpDeformableAngular3D::Sizeof - hkpJacobianSchemaInfo::DeformableAngular3D::Sizeof;
				currentAtom = atom->next();
			}
			break;

		case hkpConstraintAtom::TYPE_STIFF_SPRING:
			{
				const hkpStiffSpringConstraintAtom* atom = static_cast<const hkpStiffSpringConstraintAtom*>(currentAtom);
				atom->addToConstraintInfo(infoOut);
				infoOut.m_extraSchemaSize += hkpJacobianSchemaInfo::NpStableStiffSpring::Sizeof - hkpJacobianSchemaInfo::StableStiffSpring::Sizeof;
				currentAtom = atom->next();
			}
			break;

			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_LIN,						hkpLinConstraintAtom            );
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_LIN_SOFT,				hkpLinSoftConstraintAtom        );
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_LIN_LIMIT,				hkpLinLimitConstraintAtom       );
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_LIN_FRICTION,			hkpLinFrictionConstraintAtom	);
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_LIN_MOTOR,				hkpLinMotorConstraintAtom		);

			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_2D_ANG,					hkp2dAngConstraintAtom			);

			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_ANG,						hkpAngConstraintAtom			);
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_ANG_LIMIT,				hkpAngLimitConstraintAtom       );
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_CONE_LIMIT,				hkpConeLimitConstraintAtom		);
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_TWIST_LIMIT,				hkpTwistLimitConstraintAtom		);
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_ANG_FRICTION,			hkpAngFrictionConstraintAtom    );
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_ANG_MOTOR,				hkpAngMotorConstraintAtom       );

			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_RAGDOLL_MOTOR,			hkpRagdollMotorConstraintAtom	);
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_PULLEY,					hkpPulleyConstraintAtom			);
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_RACK_AND_PINION,			hkpRackAndPinionConstraintAtom	);
			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_COG_WHEEL,				hkpCogWheelConstraintAtom		);

			HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_WHEEL_FRICTION,			hkpWheelFrictionConstraintAtom	);

			//
			//	modifiers
			//

			// : no next() method
			//HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_MODIFIER_SOFT_CONTACT,    hkpSoftContactModifierConstraintAtom    );
			//HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_MODIFIER_MASS_CHANGER,    hkpMassChangerModifierConstraintAtom    );
			//HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_MODIFIER_VISCOUS_SURFACE, hkpViscousSurfaceModifierConstraintAtom );
			//HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE( TYPE_MODIFIER_MOVING_SURFACE,  hkpMovingSurfaceModifierConstraintAtom  );

			HK_SKIP_ATOM_BY_TYPE( TYPE_SET_LOCAL_TRANSFORMS,	hkpSetLocalTransformsConstraintAtom		);
			HK_SKIP_ATOM_BY_TYPE( TYPE_SET_LOCAL_ROTATIONS,		hkpSetLocalRotationsConstraintAtom		);
			HK_SKIP_ATOM_BY_TYPE( TYPE_SET_LOCAL_TRANSLATIONS,	hkpSetLocalTranslationsConstraintAtom	);
			HK_SKIP_ATOM_BY_TYPE( TYPE_SETUP_STABILIZATION,		hkpSetupStabilizationAtom				);

		case hkpConstraintAtom::TYPE_BRIDGE:
		case hkpConstraintAtom::TYPE_MODIFIER_SOFT_CONTACT:
		case hkpConstraintAtom::TYPE_MODIFIER_MASS_CHANGER:
		case hkpConstraintAtom::TYPE_MODIFIER_VISCOUS_SURFACE:
		case hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE:
		case hkpConstraintAtom::TYPE_MODIFIER_IGNORE_CONSTRAINT:
			{
				// this is assumed to be the last atom
				currentAtom = atomsEnd;
				HK_ASSERT2(0x74890f9d, false, "What do we do here ?");
				break;
			}

		default:
			HK_ASSERT2(0xad67de77,0,"Illegal atom.");
		}
	}
}

#undef  HK_SKIP_ATOM_BY_TYPE
#undef  HK_GET_CONSTRAINT_INFO_FROM_ATOM_BY_TYPE

void hkpConstraintData::setMaximumLinearImpulse( hkReal maxLinearImpulse )
{
	HK_WARN( 0xad809031, "setMaximumLinearImpulse() called on a constraint that doesn't support it." );
}

void hkpConstraintData::setMaximumAngularImpulse( hkReal maxAngularImpulse )
{
	HK_WARN( 0xad809032, "setMaximumAngularImpulse() called on a constraint that doesn't support it." );
}

void hkpConstraintData::setBreachImpulse( hkReal breachImpulse )
{
	
}

hkReal hkpConstraintData::getMaximumLinearImpulse() const
{
	return HK_REAL_MAX;
}

hkReal hkpConstraintData::getMaximumAngularImpulse() const
{
	return HK_REAL_MAX;
}

hkReal hkpConstraintData::getBreachImpulse() const
{
	return HK_REAL_MAX;
}

hkUint8 hkpConstraintData::getNotifiedBodyIndex() const
{
	HK_ASSERT2( 0xad809034, false, "getNotifiedBodyIndex() called on a constaint that doesn't support it." );
	return 0xff;
}

void hkpConstraintData::setBodyToNotify( int bodyIdx )
{

}

void hkpConstraintData::setSolvingMethod( hkpConstraintAtom::SolvingMethod method )
{

}

hkResult hkpConstraintData::setInertiaStabilizationFactor( const hkReal inertiaStabilizationFactor )
{
	return HK_FAILURE;
}

hkResult hkpConstraintData::getInertiaStabilizationFactor( hkReal& inertiaStabilizationFactorOut ) const
{
	return HK_FAILURE;
}

void hkpConstraintData::buildJacobian( const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out )
{
	HK_ASSERT2( 0xad567bbd, false, "Function deprecated" );
}

void hkpConstraintData::buildJacobianCallback( const hkpConstraintQueryIn &in, const hkpConstraintQueryOut& out )
{
	HK_ASSERT2( 0xad567bbd, false, "This function must be overwritten, if the constraint requires a callback." );
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
