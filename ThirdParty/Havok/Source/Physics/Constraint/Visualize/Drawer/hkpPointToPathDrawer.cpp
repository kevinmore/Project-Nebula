/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics/Constraint/Visualize/Drawer/hkpPointToPathDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Physics/Constraint/Data/PointToPath/hkpParametricCurve.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>

void hkpPointToPathDrawer::drawConstraint(const hkpPointToPathConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{

	//const hkpPointToPathConstraintData* pToPConstraint = static_cast<const hkpPointToPathConstraintData*>(constraint->getData());

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	updateCommonParameters(localToWorldA, localToWorldB);
	m_bodyBWPivot = localToWorldB.getTranslation();

	constraintData->calcPivot( localToWorldA, m_bodyAWPivot);

	drawBodyFrames(id, tag);

	drawPivots(id, tag);

	// Draw the path
	if(constraintData->getPath() != HK_NULL)
	{
		hkTransform refConstraintToWorld;
		refConstraintToWorld.setMul(localToWorldB, constraintData->getConstraintToLocalTransform(1));

		hkpParametricCurve* curve = constraintData->getPath();

		if(curve != HK_NULL)
		{

			hkArray<hkVector4> points;
			curve->getPointsToDraw(points);

			int size;
			size = points.getSize();
			hkVector4 p1;
			hkVector4 p2;
			hkVector4 p;

			for(int i = 1; i < size; i++)
			{
				p = points[i];
				p1.setTransformedPos(refConstraintToWorld, p);
				p = points[i - 1];
				p2.setTransformedPos(refConstraintToWorld, p);
				displayHandler->displayLine(p1, p2, hkColor::rgbFromFloats(0.f, .25f, 1.f), id, tag);
			}
		}
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
