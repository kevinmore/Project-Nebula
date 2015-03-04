/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Destruction/BreakOffModifier/hknpDestructionBreakOffModifier.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>

#include <Geometry/Collide/Algorithms/Triangle/hkcdTriangleUtil.h>
#include <Geometry/Internal/Algorithms/ConvexHull/hkcdConvexHullQuad2d.h>

#include <Common/Visualize/hkDebugDisplay.h>

#define ENABLE_DEBUG_DISPLAY_CONTACTS	(0)

//
//	Estimates the contact points

hkBool32 hknpDestructionBreakOffModifier::Manifold::estimateContacts(	hkVector4Parameter vVelA, const hknpShape* HK_RESTRICT shapeA, const hkTransform& worldFromA,
																		hkVector4Parameter vVelB, const hknpShape* HK_RESTRICT shapeB, const hkTransform& worldFromB,
																		hkSimdRealParameter frameDeltaTime, hkVector4& vPenDepthOut)
{
	// Compute relative linear velocity of B w.r.t. A. Estimate the time of hit based on the distances

	hkVector4 vRelVel;			vRelVel.setSub(vVelB, vVelA);	// vB - vA
	const hkSimdReal normalVel	= vRelVel.dot<3>(m_normal);		// (vB - vA) * vN
	hkVector4 vNormalVel;		vNormalVel.setAll(normalVel);	// (vB - vA) * vN

	// Compute hit time:
	//		pA(t)		= pA(0) = constant.
	//		pB(t)		= pB(0) + vNormalVel * t
	//		pB(tHit)	= pA(tHit) = pA(0) = pB(0) + distance * normal, therefore
	//		tHit		= distance * normal / vNormalVewl
	hkVector4 hitTime;		hitTime.setDiv<HK_ACC_23_BIT, HK_DIV_SET_HIGH>(m_distances, vNormalVel);
	hkVector4 vFrameTime;	vFrameTime.setAll(frameDeltaTime);

	// Estimate penetration depth
	{
		// pd = pB(dt) - pA(dt) = pB(tHit) + pB(tRest) - pA(0) = pB(dt - tHit) = vNormalVel * (dt - tHit)
		hkVector4 vTimeLeft;	vTimeLeft.setSub(vFrameTime, hitTime);
		hkVector4 vPenDepth;	vPenDepth.setMul(normalVel, vTimeLeft);

		// Negative penetration depths are not considered, as bodies already are moving apart!
		vPenDepthOut.setZero();
		vPenDepthOut.setMax(vPenDepthOut, vPenDepth);
		if ( vPenDepthOut.lessEqualZero().allAreSet() )
		{
			return false;
		}
	}

	// Only interested in hit times in this frame
	{
		// Compute manifold points at t = 0
		hkFourTransposedPoints pointsB0;	pointsB0.set(m_positions[0], m_positions[1], m_positions[2], m_positions[3]);	// Points on B at t = 0
		hkFourTransposedPoints pointsA0;	pointsA0.setAddMul(pointsB0, m_normal, m_distances);							// Points on A at t = 0

		// Compute contact points starting from A and B
		hkFourTransposedPoints contactsOnAFromB;	contactsOnAFromB.setAddMul(pointsB0, vRelVel, hitTime);	// B(0) + (vB - vA) * tHit
		hkFourTransposedPoints contactsOnBFromA;	contactsOnBFromA.setSubMul(pointsA0, vRelVel, hitTime);	// A(0) + (vA - vB) * tHit

		// The contact points will match in theory, but actually they can be quite far apart because the manifold normal is not along the relative velocity direction!
		// Pick contact points as the closest estimates to the Aabbs of A or B
		hkAabb aabbA;			shapeA->calcAabb(worldFromA, aabbA);
		hkAabb aabbB;			shapeB->calcAabb(worldFromB, aabbB);
		hkVector4 distSqA;		contactsOnAFromB.computeAabbDistanceSq(aabbA, distSqA);
		hkVector4 distSqB;		contactsOnBFromA.computeAabbDistanceSq(aabbB, distSqB);

		// Ignore manifolds where distances are greater than 5% of the Aabb size
		{
			hkVector4 extentsA;		aabbA.getExtents(extentsA);
			hkVector4 extentsB;		aabbB.getExtents(extentsB);
			const hkSimdReal tol	= hkSimdReal::fromFloat(0.05f);	// 5% of the largest extent!
			hkVector4 vTolA;		vTolA.setAll(tol * extentsA.horizontalMax<3>());
			hkVector4 vTolB;		vTolB.setAll(tol * extentsB.horizontalMax<3>());
									vTolA.mul(vTolA);
									vTolB.mul(vTolB);

			const hkVector4Comparison cmpA = distSqA.less(vTolA);
			const hkVector4Comparison cmpB = distSqB.less(vTolB);
			hkVector4Comparison cmpAB;	cmpAB.setOr(cmpA, cmpB);
			if ( !cmpAB.anyIsSet(hkVector4ComparisonMask::MASK_XYZ) )
			{
				return false;	// All contact points are too far from the actual bodies!
			}
		}

		// Compute the contacts in world
		contactsOnAFromB.setAddMul(pointsB0, vVelB, hitTime);
		contactsOnBFromA.setAddMul(pointsA0, vVelA, hitTime);

		// Select the best contact point estimates
		hkFourTransposedPoints contacts;
		const hkVector4Comparison cmpAvsB = distSqA.less(distSqB);
		contacts.setSelect(cmpAvsB, contactsOnAFromB, contactsOnBFromA);

		// Debug!
#if ( ENABLE_DEBUG_DISPLAY_CONTACTS )
		{
			const int cmpMask = cmpAvsB.getMask();
			hkVector4 cp[4];	contacts.extract(cp[0], cp[1], cp[2], cp[3]);


			for (int i = 0; i < 4; i++)
			{
				const hkVector4Comparison::Mask componentMask = (hkVector4Comparison::Mask)(hkVector4ComparisonMask::MASK_X << i);

				if ( cmpMask & componentMask )
				{
					// A from B
					hkVector4 vDir;	vDir.setSub(cp[i], m_positions[i]);
					HK_DISPLAY_ARROW(m_positions[i], vDir, 0xFFFFFFFF);
				}
				else
				{
					hkVector4 vA;	vA.setAddMul(m_positions[i], m_normal, m_distances.getComponent(i));
					hkVector4 vDir;	vDir.setSub(cp[i], vA);
					HK_DISPLAY_ARROW(vA, vDir, 0xFFFFFFFF);
				}
			}
		}
#endif

		// Save contact point
		const hkSimdReal tmpW = m_contact.getComponent<3>();
		contacts.computeAverage(m_contact);
		m_contact.setComponent<3>(tmpW);
	}

	return true;
}

//
//	Re-shuffles the manifold vertices so they form a 2D convex hull.

void hknpDestructionBreakOffModifier::Manifold::convexify()
{
	// Get source vertices and compute convex hull
	int remap[4];
	hkcdConvexHullQuad2d(m_positions, remap);

	// Compute normal
	hkVector4 vN;
	hkcdTriangleUtil::calcNonUnitNormal(m_positions[remap[0]], m_positions[remap[1]], m_positions[remap[2]], vN);

	// Combine remap table into a single byte
	const hkUint8 directCode	= (hkUint8)((remap[0] << 6) | (remap[1] << 4) | (remap[2] << 2) | remap[3]);	// A, B, C, D
	const hkUint8 flippedCode	= (hkUint8)((remap[3] << 6) | (remap[2] << 4) | (remap[1] << 2) | remap[0]);	// D, C, B, A
	const hkUint8 remapCode		= vN.dot<3>(m_normal).isLessZero() ? flippedCode : directCode;

	// Save remap code in m_contact
	m_contact.setInt24W(remapCode);
}

//
//	Computes the force that needs to be applied to body A to fix the penetrations.
//	Let this body be body A and the other body be body B. We want to apply an impulse P to body A such that the penetration depth is zero.
//		mA * vA + P = mA * vA'
//		mA * vB - P = mB * vB'
//		P = (vA' - vA) / mA = -(vB' - vB) / mB
//		mA * vA + mB * vB = mA * vA' + mB * vB'
//		vB' = mA * (vA - vA') / mB + vB
//		mAB = (mA * mB) / (mA + mB)
//
//	We want:
//		penDepth(t) = pointB(t) - pointA(t) = pointB(0) - pointA(0) + (vB' - vA') * dt = 0
//		vA' - vB' = penDepth(0) / dt = penVel
//
//	Therefore:
//		mB * vA' - mA * (vA - vA') - mB * vB = mB * penVel
//		vA' = [mB * penVel + (mA * vA + mB * vB)] / (mA + mB)
//		P = mAB * (penVel + (vB - vA))

void hknpDestructionBreakOffModifier::Manifold::computePenetrationImpulse(	hkVector4Parameter vLinearVelA, hkSimdRealParameter invMassA,
																			hkVector4Parameter vLinearVelB, hkSimdRealParameter invMassB,
																			hkVector4Parameter vPenDepth, hkSimdRealParameter invDeltaTime)
{
	// Get initial velocities
	const hkSimdReal vA = vLinearVelA.dot<3>(m_normal);
	const hkSimdReal vB = vLinearVelB.dot<3>(m_normal);
	hkSimdReal vBA;		vBA.setSub(vB, vA);

	// Compute equivalent mass
	hkSimdReal invMassAB;	invMassAB.setAdd(invMassA, invMassB);						// (mA + mB) / (mA * mB)
	hkSimdReal mAB;			mAB.setReciprocal<HK_ACC_FULL, HK_DIV_SET_HIGH>(invMassAB);	// (mA * mB) / (mA + mB)

	// Get time-step and compute penetration velocity
	hkVector4 penVel;	penVel.setAll(vBA);								// (vB - vA)
	penVel.addMul(invDeltaTime, vPenDepth);								// (penVel + (vB - vA))
	penVel.mul(mAB);

	// Keep only the maximum (signed) impulse applied on A along the manifold normal
	{
		hkVector4 xyzw			= penVel;														// [x, y, z, w]
		hkVector4 abs_xyzw;		abs_xyzw.setAbs(xyzw);											// [|x|, |y|, |z|, |w|]
		hkVector4 yxwz;			yxwz.setPermutation<hkVectorPermutation::YXWZ>(xyzw);			// [y, x, w, z]
		hkVector4 abs_yxwz;		abs_yxwz.setPermutation<hkVectorPermutation::YXWZ>(abs_xyzw);	// [|y|, |x|, |w|, |z|]

		xyzw.setSelect(abs_xyzw.greater(abs_yxwz), xyzw, yxwz);							// [signed_max(x, y), signed_max(x, y), signed_max(z, w), signed_max(z, w)]
		abs_xyzw.setMax(abs_xyzw, abs_yxwz);											// [max(|x|, |y|), max(|x|, |y|), max(|z|, |w|), max(|z|, |w|)]
		yxwz.setPermutation<hkVectorPermutation::ZWXY>(xyzw);							// [signed_max(z, w), signed_max(z, w), signed_max(x, y), signed_max(x, y)]
		abs_yxwz.setPermutation<hkVectorPermutation::ZWXY>(abs_xyzw);					// [max(|z|, |w|), max(|z|, |w|), max(|x|, |y|), max(|x|, |y|)]
		penVel.setSelect(abs_xyzw.greater(abs_yxwz), xyzw, yxwz);						// [signed_max(x, y, z, w), signed_max(x, y, z, w), signed_max(x, y, z, w), signed_max(x, y, z, w)]
	}

	// Return equivalent mass in the .w component, as we'll need it later!
	m_penImpulse.setMul(penVel.getComponent<0>(), m_normal);
	m_penImpulse.setComponent<3>(mAB);
}


//
//	Constructor

hknpDestructionBreakOffModifier::hknpDestructionBreakOffModifier()
:	hknpModifier()
{}

//
//	Destructor

hknpDestructionBreakOffModifier::~hknpDestructionBreakOffModifier()
{}

//
//	Called to process the given manifold

void hknpDestructionBreakOffModifier::manifoldProcessCallback(	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
																const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
																hknpManifold* HK_RESTRICT manifold)
{
	// Check for manifolds we should ignore
	hknpManifoldCollisionCache* manifoldCache = manifold->m_collisionCache;
	if( manifoldCache && (manifoldCache->m_bodyAndMaterialFlags & hknpBody::DONT_BUILD_CONTACT_JACOBIANS) )
	{
		return;
	}

	// Will just forward the contact manifold if at least one of the bodies is breakable
	const hknpBodyId rbIdA = cdBodyA.m_body->m_id;
	const hknpBodyId rbIdB = cdBodyB.m_body->m_id;

	// Create our manifold
	Manifold m;
	m.copy(*manifold);

	// Estimate contact points from the manifold. Ignore manifold if contact points are too far from the bodies or the bodies are separating.
	const hknpSolverInfo* solverInfo		= sharedData.m_solverInfo;
	const hknpMotion* HK_RESTRICT motionA	= cdBodyA.m_motion;
	const hknpMotion* HK_RESTRICT motionB	= cdBodyB.m_motion;

	hkVector4 vPenDepth;
	if ( m.estimateContacts(motionA->m_linearVelocity, cdBodyA.m_rootShape, cdBodyA.m_body->getTransform(),
							motionB->m_linearVelocity, cdBodyB.m_rootShape, cdBodyB.m_body->getTransform(),
							solverInfo->m_deltaTime, vPenDepth) )
	{
		m.m_bodies[0]	= rbIdA;
		m.m_bodies[1]	= rbIdB;
		m.m_materialA	= cdBodyA.m_material;
		HK_ASSERT(0x5912af41, cdBodyB.m_material == manifold->m_materialB);

		// Convexify manifold and compute penetration impulse
		m.convexify();
		m.computePenetrationImpulse(motionA->m_linearVelocity, motionA->getInverseMass(),
									motionB->m_linearVelocity, motionB->getInverseMass(),
									vPenDepth, solverInfo->m_invDeltaTime);

		// Send command
		{
			hkCommand* cmdPtr	= tl.beginCommand(sizeof(ContactEvent));
			ContactEvent* evt	= new (cmdPtr) ContactEvent();
			evt->m_manifold		= m;
			tl.endCommand(cmdPtr);
		}

		// We let the narrow-phase create the Jacobian, we'll just disable it later on if we need to
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
