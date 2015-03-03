/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>

////////////////////////////////////////////////////////////////

hkpPrimitiveDrawer::hkpPrimitiveDrawer()
{
	m_displayHandler = HK_NULL;
}

////////////////////////////////////////////////////////////////

void hkpPrimitiveDrawer::setDisplayHandler(hkDebugDisplayHandler* displayHandler)
{
	m_displayHandler = displayHandler;
}

////////////////////////////////////////////////////////////////

void hkpPrimitiveDrawer::displayArrow(const hkVector4& startPos, const hkVector4& arrowDirection, const hkVector4& perpDirection, hkColor::Argb color, hkReal scale, int id, int tag)
{
	hkVector4 endPos = startPos;
	endPos.addMul(hkSimdReal::fromFloat(scale), arrowDirection);
	m_displayHandler->displayLine(startPos, endPos, color, id, tag);
	hkQuaternion FortyFiveAboutPerpDirection; FortyFiveAboutPerpDirection.setAxisAngle(perpDirection, HK_REAL_PI/4.f);
	hkQuaternion MinusNinetyAboutPerpDirection; MinusNinetyAboutPerpDirection.setAxisAngle(perpDirection, -1.f*HK_REAL_PI/2.f);

	hkVector4 headDirection = arrowDirection;
	headDirection.setRotatedDir(FortyFiveAboutPerpDirection, headDirection);
	hkVector4 temp = endPos;
	temp.addMul(hkSimdReal::fromFloat(-scale * 0.333f), headDirection);
	m_displayHandler->displayLine(endPos, temp, color, id, tag);

	headDirection.setRotatedDir(MinusNinetyAboutPerpDirection, headDirection);
	temp = endPos;
	temp.addMul(hkSimdReal::fromFloat(-scale * 0.333f), headDirection);
	m_displayHandler->displayLine(endPos, temp, color, id, tag);
}

/////////////////////////////////////////////////////////////////

void hkpPrimitiveDrawer::drawSemiCircle(const hkVector4& center, hkVector4& normal,
								hkVector4& startPerp,hkReal thetaMin, hkReal thetaMax,
								hkReal radius, int numSegments, hkColor::Argb color, int id, int tag)
{
	hkReal thetaIncr = (thetaMax-thetaMin) / numSegments;
	hkVector4 startPos;
	startPerp.normalize<3>();
	normal.normalize<3>();

	hkRotation rot;
	rot.setAxisAngle(normal, thetaMin);

		// Initialise start position
	startPos.setMul(hkSimdReal::fromFloat(radius), startPerp);
	startPos.setRotatedDir(rot, startPos);
	startPos.add(center);

	rot.setAxisAngle(normal,thetaIncr);

		// Sucessively rotate position to next position (there will be a small amount of numerical drift here)
	for (int i = 0; i < numSegments; i++)
	{
		hkVector4 next;
		next = startPos;
		next.sub(center);
		next.setRotatedDir(rot, next);
		next.add(center);
		m_displayHandler->displayLine(startPos, next, color, id, tag);

		startPos = next;
	}
}

///////////////////////////////////////////////////////////////////////////////

void hkpPrimitiveDrawer::displayOrientedPoint(const hkVector4& position,const hkRotation& rot,
											  hkReal s, hkColor::Argb color, int id, int tag)
{
	hkVector4 p1,p2,x,y,z,scaled;

	x = rot.getColumn<0>();
	y = rot.getColumn<1>();
	z = rot.getColumn<2>();

	hkSimdReal size = hkSimdReal::fromFloat(s);
	scaled.setMul(size,x);
	p1.setSub(position,scaled);
	scaled.setMul(-size,x);
	p2.setSub(position,scaled);
	m_displayHandler->displayLine(p1, p2, color, id, tag);
	scaled.setMul(size,y);
	p1.setSub(position,scaled);
	scaled.setMul(-size,y);
	p2.setSub(position,scaled);
	m_displayHandler->displayLine(p1, p2, color, id, tag);
	scaled.setMul(size,z);
	p1.setSub(position,scaled);
	scaled.setMul(-size,z);
	p2.setSub(position,scaled);
	m_displayHandler->displayLine(p1, p2, color, id, tag);
}

////////////////////////////////////////////////////////////////////////////////

void hkpPrimitiveDrawer::displayCone(hkReal coneAngle, const hkVector4& startPos,
							 const hkVector4& coneAxis, const hkVector4& perpVector,
							 int numSegments, hkColor::Argb color, hkReal coneSize, int id, int tag)
{
	hkVector4* cone_points = hkAllocate<hkVector4>(numSegments, HK_MEMORY_CLASS_TOOLS);
	//hkReal cos_angle = hkMath::cos(coneAngle);
	hkReal segment_length = coneSize/coneAngle;
	hkQuaternion RotationFromNormal; RotationFromNormal.setAxisAngle(perpVector, coneAngle);
	hkQuaternion RotationAboutAxis; RotationAboutAxis.setAxisAngle(coneAxis, (HK_REAL_PI*2)/numSegments);

	hkVector4 offset_direction;
	offset_direction.setRotatedDir(RotationFromNormal, coneAxis);

	int i;
	for (i = 0; i < numSegments; i++)
	{
		cone_points[i] = startPos;
		cone_points[i].addMul(hkSimdReal::fromFloat(segment_length), offset_direction);
		offset_direction.setRotatedDir(RotationAboutAxis, offset_direction);
	}

	for (i = 0; i < numSegments; i++)
	{
		int next_point_index = (i+1)%numSegments;
		m_displayHandler->displayLine(startPos, cone_points[i], color, id, tag);
		m_displayHandler->displayLine(cone_points[i], cone_points[next_point_index], color, id, tag);
	}

	hkDeallocate<hkVector4>(cone_points);
}

/////////////////////////////////////////////////////////////////////////////////////////////

void hkpPrimitiveDrawer::displayPlane(const hkVector4& startPos, const hkVector4& planeNormal,
									 const hkVector4& vectorOnPlane, hkColor::Argb color, hkReal scale, int id, int tag)
{
	hkVector4 plane_points[4];
	hkReal invRoot2 = 0.70711f;
	hkVector4 newVectorOnPlane;
	newVectorOnPlane.setMul(hkSimdReal::fromFloat(scale * invRoot2), vectorOnPlane);
	hkQuaternion NinetyAboutPlaneNormal; NinetyAboutPlaneNormal.setAxisAngle(planeNormal, HK_REAL_PI/2.f);
	hkQuaternion FortyFiveAboutPlaneNormal; FortyFiveAboutPlaneNormal.setAxisAngle(planeNormal,  HK_REAL_PI/4.f);// this is only in here so that the
	//display looks more like a square than a diamond
	newVectorOnPlane.setRotatedDir(FortyFiveAboutPlaneNormal, newVectorOnPlane);

	for (int i = 0; i < 4; i++)
	{
		plane_points[i] = startPos;
		plane_points[i].add( newVectorOnPlane );
		newVectorOnPlane.setRotatedDir( NinetyAboutPlaneNormal, newVectorOnPlane);
	}

	m_displayHandler->displayLine(plane_points[0], plane_points[1], color, id, tag);
	m_displayHandler->displayLine(plane_points[1], plane_points[2], color, id, tag);
	m_displayHandler->displayLine(plane_points[2], plane_points[3], color, id, tag);
	m_displayHandler->displayLine(plane_points[3], plane_points[0], color, id, tag);
	m_displayHandler->displayLine(plane_points[0], plane_points[2], color, id, tag);
	m_displayHandler->displayLine(plane_points[1], plane_points[3], color, id, tag);

	// show the plane normal
	newVectorOnPlane.normalize<3>();
	displayArrow(startPos, planeNormal, newVectorOnPlane, color, scale, id, tag);
}

////////////////////////////////////////////////////////////////////////

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
