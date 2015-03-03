/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSceneData.h>
#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSystemDataUtil.h>

#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>

//
//	Scales all the rigid bodies in the given physics system data

void HK_CALL hknpPhysicsSystemDataUtil::scaleSystem(hknpPhysicsSceneData* sceneData, hknpPhysicsSystemData* systemData, hkSimdRealParameter uniformScale, hkArray<ShapePair>& doneShapes)
{
	// Scale the rigid bodies
	{
		hkSimdReal s_2;			s_2.setMul(uniformScale, uniformScale);
		hkSimdReal s_3;			s_3.setMul(uniformScale, s_2);
		hkSimdReal s_5;			s_5.setMul(s_2, s_3);
								s_3.setReciprocal(s_3);
								s_5.setReciprocal(s_5);
		const int numBodies		= systemData->m_bodyCinfos.getSize();

		for (int i = 0; i < numBodies; i++)
		{
			hknpBodyCinfo& rb = systemData->m_bodyCinfos[i];

			// Scale the shape
			const hknpShape* srcShape = rb.m_shape;
			if ( srcShape )
			{
				hkVector4 vScale;		vScale.setAll(uniformScale);
				hknpShape* scaledShape	= hknpShapeScalingUtil::scaleShape(srcShape, vScale, &doneShapes);

				if ( scaledShape )
				{
					rb.m_shape = scaledShape;
					sceneData->tryRemoveShape(srcShape);

					if ( systemData->m_referencedObjects.indexOf(scaledShape) < 0 )
					{
						systemData->m_referencedObjects.pushBack(scaledShape);
					}
				}
			}

			// Scale the mass properties
			if ( !rb.m_flags.anyIsSet(hknpBody::IS_KEYFRAMED | hknpBody::IS_STATIC) )
			{
				hknpMotionCinfo& motion	= systemData->m_motionCinfos[rb.m_motionId.value()];

				motion.m_inverseMass *= s_3.getReal();
				motion.m_inverseInertiaLocal.mul(s_5);
				motion.m_centerOfMassWorld.mul(uniformScale);
			}

			// Scale the position of the body about the origin
			rb.m_position.mul(uniformScale);
		}
	}

	// Scale the constraints
	{
		hkArray<hknpConstraintCinfo>& constraints = systemData->m_constraintCinfos;

		for (int i = 0; i < constraints.getSize(); i++)
		{
			hknpConstraintCinfo& constraint = constraints[i];

			// We may need to recurse in the case of breakable constraints
			bool recurse;
			hkpConstraintData* currentConstraintData = constraint.m_constraintData;

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
						data->m_atoms.m_transforms.m_transformA.getTranslation().mul( uniformScale );
						data->m_atoms.m_transforms.m_transformB.getTranslation().mul( uniformScale );
						break;
					}

				case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
					{
						// Access the limited hinge constraint data
						hkpLimitedHingeConstraintData* data = static_cast< hkpLimitedHingeConstraintData* >( currentConstraintData );

						// The local space positions in each body scale simply
						data->m_atoms.m_transforms.m_transformA.getTranslation().mul( uniformScale );
						data->m_atoms.m_transforms.m_transformB.getTranslation().mul( uniformScale );
						break;
					}

				case hkpConstraintData::CONSTRAINT_TYPE_FIXED:
					{
						hkpFixedConstraintData* data = static_cast<hkpFixedConstraintData*>(currentConstraintData);

						// Just scale the pivots' positions
						hkpSetLocalTransformsConstraintAtom& transformsAtom = data->m_atoms.m_transforms;
						transformsAtom.m_transformA.getTranslation().mul(uniformScale);
						transformsAtom.m_transformB.getTranslation().mul(uniformScale);
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

//
//	Scales all the rigid bodies in the given physics system data

void HK_CALL hknpPhysicsSystemDataUtil::scaleScene(hknpPhysicsSceneData* physicsData, hkSimdRealParameter uniformScale, hkArray<ShapePair>* doneShapes)
{
	// Cache for shape scaling
	hkArray<ShapePair> localDoneShapes;
	if ( !doneShapes )
	{
		doneShapes = &localDoneShapes;
	}

	hkArray<hknpPhysicsSystemData*> scaledSystems;
	const int numInstances =  physicsData->m_systemDatas.getSize();
	for (int si = numInstances - 1; si >= 0; si--)
	{
		hknpPhysicsSystemData* sysData = physicsData->m_systemDatas[si];

		// See if we've already scaled this system data
		if ( scaledSystems.indexOf(sysData) < 0 )
		{
			scaleSystem(physicsData, sysData, uniformScale, *doneShapes);
			scaledSystems.pushBack(sysData);
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
