/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpRagdollLimitsDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>


////////////////////////////////////////////////////////////////////

// Nearly exact copy of the code from hkpRagdollDrawer. (With exception of the pivot vectors being zeroed.)
void hkpRagdollLimitsDrawer::drawConstraint(const hkpRagdollLimitsData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	//const hkpRagdollLimitsData* ragdoll = static_cast<const hkpRagdollLimitsData*>(constraint->getData());

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, hkVector4::getZero());
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, hkVector4::getZero());
	}

	drawPivots(id, tag);

	const hkVector4* baseA = &constraintData->m_atoms.m_rotations.m_rotationA.getColumn(0);
	const hkVector4* baseB = &constraintData->m_atoms.m_rotations.m_rotationB.getColumn(0);

	// Get parameters from ragdoll constraint
	hkVector4 refTwistAxisWorld;
	refTwistAxisWorld.setRotatedDir(m_RB, baseB[hkpRagdollLimitsData::Atoms::AXIS_TWIST]);

	hkVector4 refCrossedWorld;
	refCrossedWorld.setCross( baseB[hkpRagdollLimitsData::Atoms::AXIS_TWIST], baseB[hkpRagdollLimitsData::Atoms::AXIS_PLANES] );
	refCrossedWorld.normalize<3>();
	refCrossedWorld.setRotatedDir(m_RB, refCrossedWorld);

	hkVector4 refPlaneNormalWorld;
	refPlaneNormalWorld.setRotatedDir(m_RB, baseB[hkpRagdollLimitsData::Atoms::AXIS_PLANES]);

	hkVector4 cross;
	cross.setCross(baseA[hkpRagdollLimitsData::Atoms::AXIS_PLANES], baseA[hkpRagdollLimitsData::Atoms::AXIS_TWIST]);
	cross.normalize<3>();
	hkVector4 attCrossedWorld;
	attCrossedWorld.setRotatedDir(m_RA, cross);

	hkVector4 attTwistAxisWorld;
	attTwistAxisWorld.setRotatedDir(m_RA, baseA[hkpRagdollLimitsData::Atoms::AXIS_TWIST]);

	/////////////////////////////////////////////

	//always display twist axes
	m_primitiveDrawer.displayArrow(m_bodyBWPivot, refTwistAxisWorld, refCrossedWorld, hkColor::GREEN,getArrowSizeForDraw(), id, tag);
	m_primitiveDrawer.displayArrow(m_bodyAWPivot, attTwistAxisWorld, attCrossedWorld, hkColor::rgbFromChars(255, 255, 0), getArrowSizeForDraw(), id, tag);

	// draw twist cone
	m_twistCone.setParameters( constraintData->getConeAngularLimit(), getConeSizeForDraw(), getNumArcSegmentsForDraw(), refTwistAxisWorld, m_bodyBWPivot);

	// draw plane
	hkVector4 ext; ext.setZero(); ext.setXYZ( getPlaneSizeForDraw() );
	m_plane.setParameters(refPlaneNormalWorld, refCrossedWorld, m_bodyBWPivot, ext);



	// draw plane cones
	{
		hkReal planeMaxLimit = constraintData->m_atoms.m_planesLimit.m_maxAngle;
		hkReal planeMinLimit = constraintData->m_atoms.m_planesLimit.m_minAngle;

		// These limits are negative for "lower cone", positive for upper,
		// but are "negative" cones (ie. areas in which the constrained bject/axis
		// should not be, so we have to draw cones of internal angle PI/2 - limit.

		m_planeCone1.setParameters((HK_REAL_PI*0.5f - hkMath::fabs(planeMaxLimit)), getConeSizeForDraw(), getNumArcSegmentsForDraw(), refPlaneNormalWorld, m_bodyBWPivot);

		refPlaneNormalWorld.setNeg<4>(refPlaneNormalWorld);	// Flip axis of cone to draw min limit

		m_planeCone2.setParameters((HK_REAL_PI*0.5f - hkMath::fabs(planeMinLimit)), getConeSizeForDraw(), getNumArcSegmentsForDraw(), refPlaneNormalWorld, m_bodyBWPivot);
	}


	// Pass geometries off to display handler
	{
		hkArray<hkDisplayGeometry*> twist;
		hkArray<hkDisplayGeometry*> planeCones;
		hkArray<hkDisplayGeometry*> plane;
	    twist.setSize(1);
		plane.setSize(1);
		planeCones.setSize(2);
		twist[0] = &m_twistCone;
		planeCones[0] = &m_planeCone1;
		planeCones[1] =  &m_planeCone2;
		plane[0] = &m_plane;
		displayHandler->displayGeometry(twist, hkColor::YELLOW, id, tag);
		displayHandler->displayGeometry(planeCones, hkColor::RED, id, tag);
		displayHandler->displayGeometry(plane, hkColor::rgbFromChars(255, 0, 255), id, tag);
	}
}

////////////////////////////////////////////////////////////////////

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
