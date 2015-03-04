/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/Constraint/Drawer/hkpConstraintChainDrawer.h>
#include <Physics2012/Dynamics/Constraint/Chain/BallSocket/hkpBallSocketChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/StiffSpring/hkpStiffSpringChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/Powered/hkpPoweredChainData.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>

void hkpConstraintChainDrawer::drawConstraint(const hkpConstraintChainData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag, const hkArray<hkpEntity*>& chainedEntities )
{
	m_primitiveDrawer.setDisplayHandler(displayHandler);

	int numConstraints = chainedEntities.getSize() - 1;
	
	for (int i = 0; i < numConstraints; i++)
	{
		hkTransform refLocalToWorld;
		hkTransform attLocalToWorld;

		hkpRigidBody *refBody = reinterpret_cast<hkpRigidBody*>(chainedEntities[i+1]);
		hkpRigidBody *attBody = reinterpret_cast<hkpRigidBody*>(chainedEntities[i]);

		refLocalToWorld = refBody->getTransform();
		attLocalToWorld = attBody->getTransform();

		m_RB = refLocalToWorld.getRotation();
		m_RA = attLocalToWorld.getRotation();

		m_bodyBWPos = refLocalToWorld.getTranslation();
		m_bodyAWPos = attLocalToWorld.getTranslation();

		hkVector4 pivotInB;
		hkVector4 pivotInA;

		switch( constraintData->getType() )
		{
		case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
			pivotInB = static_cast<const hkpStiffSpringChainData*>(constraintData)->m_infos[i].m_pivotInB;
			pivotInA = static_cast<const hkpStiffSpringChainData*>(constraintData)->m_infos[i].m_pivotInA;
			break;
		case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
			pivotInB = static_cast<const hkpBallSocketChainData*>(constraintData)->m_infos[i].m_pivotInB;
			pivotInA = static_cast<const hkpBallSocketChainData*>(constraintData)->m_infos[i].m_pivotInA;
			break;
		case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:
			pivotInB = static_cast<const hkpPoweredChainData*>(constraintData)->m_infos[i].m_pivotInB;
			pivotInA = static_cast<const hkpPoweredChainData*>(constraintData)->m_infos[i].m_pivotInA;
			break;
		default:
			HK_ASSERT2(0xad6777dd, false, "Chain type not supproted by the drawer.");
		}

		m_bodyBWPivot.setTransformedPos(refLocalToWorld, pivotInB);
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, pivotInA);

		// drawing
		
		drawPivots(id, tag);
		drawBodyFrames(id, tag);
		displayHandler->displayLine(m_bodyAWPivot, m_bodyBWPivot, hkColor::RED, id, tag);
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
