/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Dynamics/ScaleSystem/hkpSystemScalingUtility.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeScaling/hkpShapeScalingUtility.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>


void hkpSystemScalingUtility::scaleSystemSimd( hkpPhysicsSystem* system, hkSimdRealParameter scale, hkArray<hkpShapeScalingUtility::ShapePair>* doneShapes )
{
	// Scale the rigid bodies
	{
		// Cache for shape scaling
		hkArray< hkpShapeScalingUtility::ShapePair > localDoneShapes;
		if ( !doneShapes )
		{
			doneShapes = &localDoneShapes;
		}
		
		hkSimdReal s_2;	s_2.setMul(scale, scale);
		hkSimdReal s_3;	s_3.setMul(scale, s_2);
		hkSimdReal s_5;	s_5.setMul(s_2, s_3);
		const hkArray< hkpRigidBody* >& rbs = system->getRigidBodies();
		for ( int i = 0; i < rbs.getSize(); i++ )
		{
			hkpRigidBody* rb = rbs[i];

			// Scale the shape
			hkpShape* shape = const_cast< hkpShape* >( rb->getCollidable()->getShape() );
			hkpShapeScalingUtility::scaleShapeSimd( shape, scale, doneShapes );

			// Scale the mass properties
			if ( !rb->isFixedOrKeyframed() )
			{
				rb->setMass( rb->getMass() * s_3.getReal() );

				hkMatrix3 inertia;
				rb->getInertiaLocal(inertia);
				inertia.mul( s_5 );
				rb->setInertiaLocal( inertia );
			}

			hkVector4 com;
			com.setMul( scale, rb->getCenterOfMassLocal() );
			rb->setCenterOfMassLocal( com );

			// Scale the position of the body about the origin
			hkVector4 pos;
			pos.setMul( scale, rb->getPosition() );
			rb->setPosition( pos );

			// Scale the local frames
			if ( rb->m_localFrame.val() != HK_NULL )
			{
				hkArray< hkLocalFrame* > lfStack;
				lfStack.pushBack( rb->m_localFrame.val() );

				while ( lfStack.getSize() )
				{
					// Pop off the top of the stack
					hkLocalFrame* lf = lfStack.back();
					lfStack.popBack();

					// Scale the translation of the transform
					hkTransform xform;
					lf->getLocalTransform( xform );
					xform.getTranslation().mul( scale );
					lf->setLocalTransform( xform );

					// Push back all children
					for ( int child = 0; child < lf->getNumChildFrames(); child++ )
					{
						lfStack.pushBack( lf->getChildFrame( child ) );
					}
				}
			}
		}
	}

	// Scale the constraints
	{
		const hkArray< hkpConstraintInstance* >& constraints = system->getConstraints();
		for ( int i = 0; i < constraints.getSize(); i++ )
		{
			hkpConstraintInstance* constraint = constraints[i];
			
			// We may need to recurse in the case of breakable constraints
			bool recurse;
			hkpConstraintData* currentConstraintData = constraint->getDataRw();

			do
			{
				recurse = false;

				const int type = currentConstraintData->getType();
				switch( type )
				{
				case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
					{
						// Access the ragdoll constraint data
						hkpRagdollConstraintData* data = static_cast< hkpRagdollConstraintData* >( currentConstraintData );

						// The local space positions in each body scale simply
						data->m_atoms.m_transforms.m_transformA.getTranslation().mul( scale );
						data->m_atoms.m_transforms.m_transformB.getTranslation().mul( scale );
						break;
					}

				case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
					{
						// Access the limited hinge constraint data
						hkpLimitedHingeConstraintData* data = static_cast< hkpLimitedHingeConstraintData* >( currentConstraintData );

						// The local space positions in each body scale simply
						data->m_atoms.m_transforms.m_transformA.getTranslation().mul( scale );
						data->m_atoms.m_transforms.m_transformB.getTranslation().mul( scale );
						break;
					}

				case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:
					{
						hkpBreakableConstraintData* data = static_cast<hkpBreakableConstraintData*>(currentConstraintData);

						// We need to scale the strength. The masses scale by f^3. For a linear constraint, the strength = impulse
						// and is computed as:
						//		strength = Inverse(K) * err
						//	K	= (1/mA) + (1/mB) + Hat(armA) * invInertiaA * Hat(armA) + Hat(armB) * invInertiaB * Hat(armB)
						//		= (1/f)^3 * (1/mA0 + 1/mB0) + (f^2) * Hat(armA0) * invInertiaA0/(f^5) * Hat(armA) +...
						//		= (1/f)^3 * (1/mA0 + 1/mB0 + Hat(armA0) * invInertiaA0 * Hat(armA0) + Hat(armB0) * invInertiaB0 * Hat(armB0))
						//		= (1/f)^3 * K0
						// So we need to scale the strength by f^3
						{
							hkSimdReal f_3;
							f_3.setMul(scale, scale);
							f_3.mul(scale);
							data->m_solverResultLimit *= f_3.getReal();
						}

						// We need to recurse in order to process the embedded constraint data
						currentConstraintData = data->m_constraintData;
						recurse = true;
					}
					break;

				case hkpConstraintData::CONSTRAINT_TYPE_FIXED:
					{
						hkpFixedConstraintData* data = static_cast<hkpFixedConstraintData*>(currentConstraintData);

						// Just scale the pivots' positions
						hkpSetLocalTransformsConstraintAtom& transformsAtom = data->m_atoms.m_transforms;
						transformsAtom.m_transformA.getTranslation().mul(scale);
						transformsAtom.m_transformB.getTranslation().mul(scale);
					}
					break;

				default:
					{
						HK_ASSERT2( 0x2EA1308A, false, "Only ragdoll and limited hinge constraints are supported for scaling.");
						break;
					}
				}
			} while ( recurse );
		}
	}
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
