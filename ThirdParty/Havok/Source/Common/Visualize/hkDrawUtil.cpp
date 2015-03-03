/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#define HK_USE_DEBUG_DISPLAY
#include <Common/Visualize/hkDrawUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/Shape/hkDisplayBox.h>
#include <Common/Visualize/Shape/hkDisplaySemiCircle.h>
#include <Geometry/Collide/Algorithms/Triangle/hkcdTriangleUtil.h>

hkDrawUtil::hkDrawUtil(hkReal new_scale):
	m_scale(new_scale)
{
}


void hkDrawUtil::displayOrientedPoint(const hkVector4& position,const hkRotation& rot,
									  hkReal s,hkColor::Argb color)
{
	hkVector4 p1,p2,x,y,z,scaled;

	x = rot.getColumn(0);
	y = rot.getColumn(1);
	z = rot.getColumn(2);

	hkSimdReal size; size.setFromFloat(s);
	scaled.setMul(size,x);
	p1.setSub(position,scaled);
	scaled.setMul(-size,x);
	p2.setSub(position,scaled);
	HK_DISPLAY_LINE(p1, p2, color);
	scaled.setMul(size,y);
	p1.setSub(position,scaled);
	scaled.setMul(-size,y);
	p2.setSub(position,scaled);
	HK_DISPLAY_LINE(p1, p2, color);
	scaled.setMul(size,z);
	p1.setSub(position,scaled);
	scaled.setMul(-size,z);
	p2.setSub(position,scaled);
	HK_DISPLAY_LINE(p1, p2, color);
}

	

void hkDrawUtil::displayPoint(const hkVector4& position,hkReal size,hkColor::Argb color)
{
	hkVector4 p1,p2;
	p1 = position;
	p2 = position;
	p1(0)-=size;
	p2(0)+=size;
	HK_DISPLAY_LINE(p1, p2, color);
	p1(0) = position(0);
	p2(0) = position(0);
	p1(1)-=size;
	p2(1)+=size;
	HK_DISPLAY_LINE(p1, p2, color);
	p1(1) = position(1);
	p2(1) = position(1);
	p1(2)-=size;
	p2(2)+=size;
	HK_DISPLAY_LINE(p1, p2, color);
}

void hkDrawUtil::displaySegment(const hkVector4& p1, const hkVector4& p2, hkColor::Argb color)
{
	HK_DISPLAY_LINE(p1, p2, color);
}

hkReal hkDrawUtil::getScale()
{
	return m_scale;
}

void hkDrawUtil::setScale(hkReal newScale)
{
	m_scale = newScale;
}

void hkDrawUtil::displayCone(hkReal coneAngle, const hkVector4& startPos, 
							 const hkVector4& coneAxis, const hkVector4& perpVector, 
							 int numSegments, hkColor::Argb color)
{
	hkVector4* cone_points = hkAlignedAllocate<hkVector4>(HK_REAL_ALIGNMENT,numSegments, HK_MEMORY_CLASS_TOOLS);
	hkReal segment_length = m_scale/coneAngle;
	hkQuaternion RotationFromNormal; RotationFromNormal.setAxisAngle(perpVector, coneAngle);
	hkQuaternion RotationAboutAxis; RotationAboutAxis.setAxisAngle(coneAxis, (HK_REAL_PI*2)/numSegments);

	hkVector4 offset_direction;
	offset_direction.setRotatedDir(RotationFromNormal, coneAxis);

	int i;
	for (i = 0; i < numSegments; i ++)
	{
		cone_points[i] = startPos;
		hkSimdReal segLen; segLen.setFromFloat(segment_length);
		cone_points[i].addMul(segLen, offset_direction);
		offset_direction.setRotatedDir(RotationAboutAxis, offset_direction);
	}

	for (i = 0; i < numSegments; i++)
	{
		int next_point_index = (i+1)%numSegments;
		displaySegment(startPos, cone_points[i],color);
		displaySegment(cone_points[i], cone_points[next_point_index],color);
	}
	hkAlignedDeallocate<hkVector4>(cone_points);
}


void hkDrawUtil::drawSemiCircle(const hkVector4& center, hkVector4& normal, 
								hkVector4& startPerp,hkReal thetaMin,hkReal thetaMax,
								hkReal radius,int numSegments, 
								hkColor::Argb color)
{
	hkReal thetaIncr = (thetaMax-thetaMin) / numSegments;
	hkVector4 startPos;
	startPerp.normalize<3>();
	normal.normalize<3>();

	hkRotation rot;
	rot.setAxisAngle(normal, thetaMin);

		// Initialise start position
	hkSimdReal sradius; sradius.setFromFloat(radius);
	startPos.setMul(sradius, startPerp);	
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
		displaySegment(startPos, next, color);

		startPos = next;
	}
}


void hkDrawUtil::displayPlane(const hkVector4& startPos, const hkVector4& planeNormal, 
							  const hkVector4& vectorOnPlane, hkColor::Argb color, hkBool showNormal)
{
	
	hkVector4 plane_points[4];
	hkReal invRoot2 = 0.70711f; 
	hkVector4 newVectorOnPlane;
	hkSimdReal v; v.setFromFloat(m_scale * invRoot2);
	newVectorOnPlane.setMul(v, vectorOnPlane);
	hkQuaternion NinetyAboutPlaneNormal; NinetyAboutPlaneNormal.setAxisAngle(planeNormal, HK_REAL_PI/2);
	hkQuaternion FortyFiveAboutPlaneNormal; FortyFiveAboutPlaneNormal.setAxisAngle(planeNormal,  HK_REAL_PI/4);// this is only in here so that the 
	//display looks more like a square than a diamond
	newVectorOnPlane.setRotatedDir(FortyFiveAboutPlaneNormal, newVectorOnPlane);
	for (int i = 0; i<4; i++)
	{
		plane_points[i] = startPos;
		plane_points[i].add( newVectorOnPlane );
		newVectorOnPlane.setRotatedDir( NinetyAboutPlaneNormal, newVectorOnPlane);
	}
	displaySegment(plane_points[0], plane_points[1],color);
	displaySegment(plane_points[1], plane_points[2],color);
	displaySegment(plane_points[2], plane_points[3],color);
	displaySegment(plane_points[3], plane_points[0],color);
	displaySegment(plane_points[0], plane_points[2],color);
	displaySegment(plane_points[1], plane_points[3],color);
	if (showNormal)
	{
		newVectorOnPlane.normalize<3>();
		displayArrow(startPos, planeNormal, newVectorOnPlane, color);
	}
	
}


void hkDrawUtil::displayArrow(const hkVector4& startPos, const hkVector4& arrowDirection, 
							  const hkVector4& perpDirection, hkColor::Argb color)
{
	hkVector4 endPos = startPos;
	hkSimdReal s; s.setFromFloat(m_scale);
	endPos.addMul(s, arrowDirection);
	displaySegment(startPos, endPos, color);
	hkQuaternion FortyFiveAboutPerpDirection; FortyFiveAboutPerpDirection.setAxisAngle(perpDirection, HK_REAL_PI/4);
	hkQuaternion MinusNinetyAboutPerpDirection; MinusNinetyAboutPerpDirection.setAxisAngle(perpDirection, -1*HK_REAL_PI/2);

	hkVector4 headDirection = arrowDirection;
	headDirection.setRotatedDir(FortyFiveAboutPerpDirection, headDirection);
	hkVector4 temp = endPos;
	hkSimdReal shortS = -s * hkSimdReal::getConstant<HK_QUADREAL_INV_3>();
	temp.addMul(shortS, headDirection);
	displaySegment(endPos, temp , color);

	headDirection.setRotatedDir(MinusNinetyAboutPerpDirection, headDirection);
	temp = endPos;
	temp.addMul(shortS, headDirection);
	displaySegment(endPos, temp, color);
	
}


void hkDrawUtil::displayMinMaxRectangleWidget( const hkTransform& locationOfBase, const hkVector4& normal, hkReal length, hkReal width, hkReal minHeight, hkReal maxHeight, const hkStringPtr& label, hkColor::Argb color, int tag)
{
	// Min and Max plane transforms are now the center points of both planes.
	hkSimdReal minH; minH.setFromFloat( minHeight );
	hkSimdReal maxH; maxH.setFromFloat( maxHeight );

	hkVector4 minPlaneCentroid;
	hkVector4 maxPlaneCentroid;
	minPlaneCentroid.setAddMul( locationOfBase.getTranslation(), normal, minH );
	maxPlaneCentroid.setAddMul( locationOfBase.getTranslation(), normal, maxH );

	// Calculate binormal and tangents of this plane
	int major = normal.getIndexOfMaxAbsComponent<3>();

	// Binormal is whatever axis is after ( 0 being after 3 ) the normal axis, but oriented orthogonally to the normal
	hkVector4 binormal; 
	binormal.setZero();
	binormal( (major + 1) % 3 ) = 1.0f;

	binormal.setCross( normal, binormal );
	binormal.normalize< 3, HK_ACC_12_BIT, HK_SQRT_SET_ZERO >();

	// And the tangent is the cross of the new binormal and the normal.
	hkVector4 tangent;
	tangent.setCross( binormal, normal );
	tangent.normalize< 3, HK_ACC_12_BIT, HK_SQRT_SET_ZERO >();

	// If we're on the one axis that doesn't align to the way we like our rectangles oriented, we swap length and width to avoid spinning.
	// This is equivalent to swapping the tangent and binormal since they're in a different type of orientation than the other two planes.
	if( major == 2 )
	{
		hkAlgorithm::swap<hkReal>( length, width );
	}

	// Now draw our rectangles
	displayOrientedRectangle( minPlaneCentroid, binormal, tangent, length, width, color );
	displayOrientedRectangle( maxPlaneCentroid, binormal, tangent, length, width, color );

	// Draw a line connecting the two planes, and label
	hkDebugDisplay::getInstance().displayLine( maxPlaneCentroid, minPlaneCentroid, color, 0, tag );
	
	// Draw the text
	if ( (label != HK_NULL) && (label != "") )
	{
		hkVector4 textLocation;
		hkSimdReal c; c.setFromFloat( 0.66f );
		textLocation.setInterpolate( minPlaneCentroid,  maxPlaneCentroid, c );
		hkDebugDisplay::getInstance().display3dText( label, textLocation, color, 0, tag );
	}
}

void hkDrawUtil::displayAxisAngleWidget( const hkVector4& location, const hkVector4& normal, const hkVector4& parent, const hkVector4& child, hkReal minRadians, hkReal maxRadians, hkReal radius, hkInt32 numSegments, hkColor::Argb color, int tag)
{
	// Draw lines representing the parent and child bones
	{
		hkDebugDisplay::getInstance().displayLine(location, parent, color, 0, tag);
		hkDebugDisplay::getInstance().displayLine(location, child, color, 0, tag);
	}

	// Draw an arrow representing the axis
	hkVector4 arrowScaledNormal;
	hkSimdReal halfRad; halfRad.setFromFloat(radius * 0.5f);
	arrowScaledNormal.setMul(halfRad, normal);
	hkDebugDisplay::getInstance().displayArrow(location, arrowScaledNormal, color, 0, tag);

	// Compute the semi-circle's perpendicular vector, which defines 0 rotation along the circle
	hkVector4 perpendicular;
	perpendicular.setSub(parent, location);
	perpendicular.normalize<3>();

	// Create and display the semicircle representing the angle range
	hkDisplaySemiCircle angleSemicircle;
	angleSemicircle.setParameters(radius, minRadians, maxRadians, numSegments, location, normal, perpendicular);
	hkArray<hkDisplayGeometry*> geometry;
	geometry.setSize(1);
	geometry[0] = &(angleSemicircle);
	hkDebugDisplay::getInstance().displayGeometry(geometry, color, 0, tag);
}

void hkDrawUtil::displayLocalFrame( const hkLocalFrame& localFrame, const hkTransform& worldFromModel, hkReal size, bool drawChildren, hkColor::Argb color, const char* annotation)
{
	hkArray<const hkLocalFrame*> descendants;
	descendants.pushBack(&localFrame);
	localFrame.getDescendants(descendants);

	hkTransform transform;

	for( int i = 0; i < descendants.getSize(); ++i )
	{		
		hkStringBuf name;

		const hkLocalFrame* childFrame = descendants[i];
		const char* childFrameName = childFrame->getName();
		if ( childFrameName != HK_NULL )
		{
			name += childFrameName;
			childFrame->getTransformToRoot(transform);

			transform.setMul(worldFromModel, transform);

			name += annotation;		
			HK_DISPLAY_3D_TEXT(name, transform.getTranslation(), color);

			HK_DISPLAY_FRAME(transform, size);
		}
	}
}

void HK_CALL hkDrawUtil::debugDisplayGeometry(const hkGeometry& triMesh, const DebugDisplayGeometrySettings& settings )
{
	if( settings.m_showVertices || settings.m_labelVertices )
	{
		const int numVertices = triMesh.m_vertices.getSize();
		for (int i = 0; i < numVertices; i++)
		{
			if(settings.m_showVertices)
			{
				HK_DISPLAY_STAR(triMesh.m_vertices[i], 0.05f, settings.m_vertexColor );
			}

			if( settings.m_labelVertices )
			{
				hkStringBuf label; 
				label.printf("%d", i);
				HK_DISPLAY_3D_TEXT( label.cString(), triMesh.m_vertices[i], settings.m_vertexColor );
			}

		}
	}

	const int skipFaces = hkMath::max2(settings.m_skipFaces, 1);

	if( settings.m_showFaces || settings.m_showEdges || settings.m_labelTriangles )
	{
		hkVector4 lightDir = settings.m_lightDirection;
		lightDir.normalize<3>();

		const int numTris = triMesh.m_triangles.getSize();

		for (int i = 0; i < numTris; i = i + skipFaces )
		{
			const hkGeometry::Triangle& tri = triMesh.m_triangles[i];

			const hkVector4& v0 = triMesh.m_vertices[tri.m_a];
			const hkVector4& v1 = triMesh.m_vertices[tri.m_b];
			const hkVector4& v2 = triMesh.m_vertices[tri.m_c];

			if( !settings.m_cullingAabb.isEmpty() )
			{
				hkAabb aabbTri;
				aabbTri.setEmpty();
				aabbTri.includePoint( v0 );
				aabbTri.includePoint( v1 );
				aabbTri.includePoint( v2 );

				if( !settings.m_cullingAabb.overlaps(aabbTri) )
					continue;

			}

			hkColor::Argb faceColor;
			if( settings.m_forceColorFacesByMaterial )
			{
				faceColor = hkColor::getPaletteColor( hkUint32(tri.m_material) );
			}
			else
			{
				faceColor = settings.m_faceColor;
			}

			if( settings.m_showEdges )
			{
				HK_DISPLAY_LINE(v0, v1, settings.m_edgeColor);
				HK_DISPLAY_LINE(v1, v2, settings.m_edgeColor);
				HK_DISPLAY_LINE(v2, v0, settings.m_edgeColor);
			}

			if ( settings.m_labelTriangles )
			{
				hkVector4 centroid; hkcdTriangleUtil::calcCentroid( v0, v1, v2, centroid );
				hkStringBuf label; label.printf("%d",i);
				HK_DISPLAY_3D_TEXT(label.cString(), centroid, faceColor);
			}

			if( settings.m_showFaces )
			{
				if (settings.m_lightFaces )
				{
					const hkReal r = hkColor::getRedAsFloat(faceColor);
					const hkReal g = hkColor::getGreenAsFloat(faceColor);
					const hkReal b = hkColor::getBlueAsFloat(faceColor);

					hkVector4 ee0;	ee0.setSub(v1, v0);
					hkVector4 ee1;	ee1.setSub(v2, v0);
					hkVector4 n;
					n.setCross(ee0, ee1);
					n.normalize<3>();
					hkReal d = n.dot<3>( lightDir ).getReal();
					d = 0.1f + 0.9f * hkMath::clamp(d, hkReal(0.0f), hkReal(1.0f) );
					faceColor = hkColor::rgbFromFloats( d*r, d*g, d*b, hkColor::getAlphaAsFloat(faceColor) );
				}

				HK_DISPLAY_TRIANGLE(v0, v1, v2, faceColor);
			}
		}
	}
}

void hkDrawUtil::displayOrientedRectangle( const hkVector4& center, const hkVector4& binormal, const hkVector4& tangent, hkReal length, hkReal width, hkColor::Argb color )
{
	// ^ 
	// |  == binormal  and --> == tangent
	// 

	// Direction vectors for navigating the rectangle.
	hkVector4 negativeTangent, negativeBinormal;
	negativeTangent.setNeg<3>( tangent );
	negativeBinormal.setNeg<3>( binormal );

	// Points of the rectangle
	hkVector4 topLeft, bottomRight, bottomLeft, topRight;

	// Get half the length and half the width for creating the rectangle
	hkSimdReal halfLength, halfWidth;
	halfLength.setFromFloat( length * 0.5f );
	halfWidth.setFromFloat( width * 0.5f );

	// Start at the center, right halfLength, and up halfWidth
	topRight.setAddMul( center, tangent, halfLength );
	topRight.setAddMul( topRight, binormal, halfWidth );

	// Top left is left halfLength, and up halfWidth
	topLeft.setAddMul( center, negativeTangent, halfLength );
	topLeft.setAddMul( topLeft, binormal, halfWidth );

	// Bottom left is left halfLength, and down halfWidth
	bottomLeft.setAddMul( center, negativeTangent, halfLength );
	bottomLeft.setAddMul( bottomLeft, negativeBinormal, halfWidth );

	// Bottom right is right halfLength, and down halfWidth
	bottomRight.setAddMul( center, tangent, halfLength );
	bottomRight.setAddMul( bottomRight, negativeBinormal, halfWidth );

	// Draw lines connecting the 4 corners of the rectangle.
	displaySegment(topRight, bottomRight, color);
	displaySegment(bottomRight, bottomLeft, color);
	displaySegment(bottomLeft, topLeft, color);
	displaySegment(topLeft, topRight, color);
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
