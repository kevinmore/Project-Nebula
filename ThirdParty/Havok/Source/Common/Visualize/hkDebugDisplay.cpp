/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

  
hkDebugDisplay::hkDebugDisplay()
{
	m_arrayLock = new hkCriticalSection(1000); // usually no contention
}

hkDebugDisplay::~hkDebugDisplay()
{
	delete m_arrayLock;
}

void hkDebugDisplay::addDebugDisplayHandler(hkDebugDisplayHandler* debugDisplay)
{
	m_arrayLock->enter();
		m_debugDisplayHandlers.pushBack( debugDisplay );
	m_arrayLock->leave();
}

void hkDebugDisplay::removeDebugDisplayHandler(hkDebugDisplayHandler* debugDisplay)
{
	m_arrayLock->enter();
	int index = m_debugDisplayHandlers.indexOf(debugDisplay);
	if(index >= 0)
	{
		m_debugDisplayHandlers.removeAt(index); 
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::clear()
{
	m_arrayLock->enter();
	m_debugDisplayHandlers.clear();
	m_arrayLock->leave();
}

//
// Debug Display functionality
//
void hkDebugDisplay::addGeometry(const hkArray<hkDisplayGeometry*>& geometries, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->addGeometry(geometries, transform, id, tag, shapeIdHint);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::addGeometry(hkDisplayGeometry* geometry, hkUlong id, int tag, hkUlong shapeIdHint)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->addGeometry(geometry, id, tag, shapeIdHint);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::addGeometryInstance(hkUlong instanceId, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->addGeometryInstance(instanceId, transform, id, tag, shapeIdHint);
	}
	m_arrayLock->leave();
}


void hkDebugDisplay::setGeometryPickable( hkBool isPickable, hkUlong id, int tag )
{
	m_arrayLock->enter();
	for( int i = 0; i < m_debugDisplayHandlers.getSize(); i++ )
	{
		m_debugDisplayHandlers[i]->setGeometryPickable( isPickable, id, tag );
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::setGeometryColor(hkColor::Argb color, hkUlong id, int tag)
{
	// keep a record of the color change (so that new displayViewer may be informed
	// NOT going to worry about this for the moment...
	m_arrayLock->enter();
	// send color change out too all existing displayViewers
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->setGeometryColor(color, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::updateGeometry(const hkTransform& transform, hkUlong id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->updateGeometry(transform, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::updateGeometry(const hkMatrix4& transform, hkUlong id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->updateGeometry(transform, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::updateGeometry( const hkQsTransform& transform, hkUlong id, int tag )
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->updateGeometry(transform, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::skinGeometry(hkUlong* ids, int numIds, const hkMatrix4* poseModel, int numPoseModel, const hkMatrix4& worldFromModel, int tag )
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->skinGeometry(ids, numIds, poseModel, numPoseModel, worldFromModel, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::skinGeometry(hkUlong* ids, int numIds, const hkQsTransform* poseModel, int numPoseModel, const hkQsTransform& worldFromModel, int tag )
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->skinGeometry(ids, numIds, poseModel, numPoseModel, worldFromModel, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::removeGeometry(hkUlong id, int tag, hkUlong shapeIdHint)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->removeGeometry(id, tag, shapeIdHint);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::updateCamera(const hkVector4& from, const hkVector4& to, const hkVector4& up, hkReal nearPlane, hkReal farPlane, hkReal fov, const char* name)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->updateCamera(from, to, up, nearPlane, farPlane, fov, name);
	}
	m_arrayLock->leave();

}

void hkDebugDisplay::displayPoint(const hkVector4& position, hkColor::Argb color, int id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->displayPoint(position, color, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::displayLine(const hkVector4& start, const hkVector4& end, hkColor::Argb color, int id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
//		m_debugDisplayHandlers[i]->holdImmediate();
		m_debugDisplayHandlers[i]->displayLine(start, end, color, id, tag);
//		m_debugDisplayHandlers[i]->step();
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::displayTriangle(const hkVector4& a, const hkVector4& b, const hkVector4& c, hkColor::Argb color, int id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->displayTriangle(a,b,c, color, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::displayLineModelSpace(const hkQsTransform& worldFromModel, const hkVector4& start, const hkVector4& end, hkColor::Argb color, int id, int tag)
{
	hkVector4 startWorld;
	hkVector4 endWorld;

	startWorld.setTransformedPos( worldFromModel, start );
	endWorld.setTransformedPos( worldFromModel, end );

	displayLine( startWorld, endWorld, color, id, tag );
}

void hkDebugDisplay::displayLineModelSpace(const hkTransform& worldFromModel, const hkVector4& start, const hkVector4& end, hkColor::Argb color, int id, int tag)
{
	hkQsTransform t;
	t.setFromTransform( worldFromModel );
	displayLineModelSpace( t, start, end, color, id, tag );
}

void hkDebugDisplay::displayRay(const hkVector4& start, const hkVector4& direction, hkColor::Argb color, int id, int tag)
{
	hkVector4 end;
	end.setAdd( start, direction );

	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->displayLine(start, end, color, id, tag);
	}
}

void hkDebugDisplay::displayRayModelSpace(const hkQsTransform& worldFromModel, const hkVector4& start, const hkVector4& direction, hkColor::Argb color, int id, int tag)
{
	hkVector4 startWorld;
	hkVector4 directionWorld;

	startWorld.setTransformedPos( worldFromModel, start );
	directionWorld.setRotatedDir( worldFromModel.getRotation(), direction );

	displayRay( startWorld, directionWorld, color, id, tag );
}

void hkDebugDisplay::displayRayModelSpace(const hkTransform& worldFromModel, const hkVector4& start, const hkVector4& direction, hkColor::Argb color, int id, int tag)
{
	hkQsTransform t;
	t.setFromTransform( worldFromModel );
	displayRayModelSpace( t, start, direction, color, id, tag );
}

void hkDebugDisplay::displayFrame( const hkTransform& worldFromLocal, hkReal size, int id, int tag )
{
	hkVector4 ZERO;
	hkVector4 X;
	hkVector4 Y;
	hkVector4 Z;

	hkVector4 vec; vec.setZero();
	ZERO.setTransformedPos( worldFromLocal, vec );
	vec.set( size, 0, 0, 0 );
	X.setTransformedPos( worldFromLocal, vec );
	vec.set( 0, size, 0, 0 );
	Y.setTransformedPos( worldFromLocal, vec );
	vec.set( 0, 0, size, 0 );
	Z.setTransformedPos( worldFromLocal, vec );

	hkVector4 dirX; dirX.setSub( X, ZERO );
	hkVector4 dirY; dirY.setSub( Y, ZERO );
	hkVector4 dirZ; dirZ.setSub( Z, ZERO );

	displayArrow( ZERO, dirX, hkColor::RED, id, tag );
	displayArrow( ZERO, dirY, hkColor::GREEN, id, tag );
	displayArrow( ZERO, dirZ, hkColor::BLUE, id, tag );
}

void hkDebugDisplay::displayFrame( const hkQsTransform& worldFromLocal, hkReal size, int id, int tag )
{
	hkVector4 ZERO;
	hkVector4 X;
	hkVector4 Y;
	hkVector4 Z;

	hkVector4 vec; vec.setZero();
	ZERO.setTransformedPos( worldFromLocal, vec );
	vec.set( size, 0, 0, 0 );
	X.setTransformedPos( worldFromLocal, vec );
	vec.set( 0, size, 0, 0 );
	Y.setTransformedPos( worldFromLocal, vec );
	vec.set( 0, 0, size, 0 );
	Z.setTransformedPos( worldFromLocal, vec );

	hkVector4 dirX; dirX.setSub( X, ZERO );
	hkVector4 dirY; dirY.setSub( Y, ZERO );
	hkVector4 dirZ; dirZ.setSub( Z, ZERO );

	displayArrow( ZERO, dirX, hkColor::RED, id, tag );
	displayArrow( ZERO, dirY, hkColor::GREEN, id, tag );
	displayArrow( ZERO, dirZ, hkColor::BLUE, id, tag );
}

void hkDebugDisplay::displayArrow(const hkVector4& from, const hkVector4& dir, hkColor::Argb color, int id, int tag)
{
	// Check that we have a valid direction
	if (dir.lengthSquared<3>().getReal() < HK_REAL_EPSILON)
	{
		return;
	}

	hkVector4 to; to.setAdd( from, dir );
	hkVector4 ort; hkVector4Util::calculatePerpendicularVector( dir, ort );
	ort.normalize<3>();
	hkVector4 ort2; ort2.setCross( dir, ort );

	ort.mul( dir.length<3>() );

	hkSimdReal c0; c0.setFromFloat(0.85f);
	hkSimdReal c; c.setFromFloat(1.0f - 0.85f);
	hkVector4 p; p.setInterpolate( from, to, c0 );
	hkVector4 p0; p0.setAddMul( p, ort, c );
	hkVector4 p1; p1.setAddMul( p, ort, -c );
	hkVector4 p2; p2.setAddMul( p, ort2, c );
	hkVector4 p3; p3.setAddMul( p, ort2, -c );

	displayLine( from, to, color, id, tag );
	displayLine( to, p0, color, id, tag );
	displayLine( to, p1, color, id, tag );
	displayLine( to, p2, color, id, tag );
	displayLine( to, p3, color, id, tag );
}

void hkDebugDisplay::displayStar(const hkVector4& position, hkReal scale, hkColor::Argb color, int id, int tag)
{
	for (int k=0; k<3; k++)
	{
		hkVector4 star, pt1, pt2;
		star.setZero();

		star(k) = scale;
		pt1.setAdd(position,star);
		pt2.setSub(position,star);
		displayLine(pt1, pt2, color, id, tag);
	}
}

void hkDebugDisplay::displayStarModelSpace(const hkQsTransform& worldFromModel, const hkVector4& position, hkReal scale, hkColor::Argb color, int id, int tag)
{
	hkVector4 positionWorld;
	positionWorld.setTransformedPos( worldFromModel, position );
	displayStar( positionWorld, scale, color, id, tag );
}

void hkDebugDisplay::displayStarModelSpace(const hkTransform& worldFromModel, const hkVector4& position, hkReal scale, hkColor::Argb color, int id, int tag)
{
	hkVector4 positionWorld;
	positionWorld.setTransformedPos( worldFromModel, position );
	displayStar( positionWorld, scale, color, id, tag );
}

void hkDebugDisplay::displayAabb(const hkAabb& aabb, hkColor::Argb color, int id, int tag)
{
	hkVector4 lines[24];

	lines[0].set(aabb.m_min(0),aabb.m_min(1),aabb.m_min(2));
	lines[1].set(aabb.m_min(0),aabb.m_max(1),aabb.m_min(2));

	lines[2].set(aabb.m_min(0),aabb.m_min(1),aabb.m_min(2));
	lines[3].set(aabb.m_min(0),aabb.m_min(1),aabb.m_max(2));

	lines[4].set(aabb.m_min(0),aabb.m_min(1),aabb.m_min(2));
	lines[5].set(aabb.m_max(0),aabb.m_min(1),aabb.m_min(2));

	lines[6].set(aabb.m_max(0),aabb.m_max(1),aabb.m_max(2));
	lines[7].set(aabb.m_max(0),aabb.m_max(1),aabb.m_min(2));

	lines[8].set(aabb.m_max(0),aabb.m_max(1),aabb.m_max(2));
	lines[9].set(aabb.m_min(0),aabb.m_max(1),aabb.m_max(2));

	lines[10].set(aabb.m_max(0),aabb.m_max(1),aabb.m_max(2));
	lines[11].set(aabb.m_max(0),aabb.m_min(1),aabb.m_max(2));

	lines[12].set(aabb.m_min(0),aabb.m_max(1),aabb.m_min(2));
	lines[13].set(aabb.m_max(0),aabb.m_max(1),aabb.m_min(2));

	lines[14].set(aabb.m_min(0),aabb.m_max(1),aabb.m_min(2));
	lines[15].set(aabb.m_min(0),aabb.m_max(1),aabb.m_max(2));

	lines[16].set(aabb.m_max(0),aabb.m_max(1),aabb.m_min(2));
	lines[17].set(aabb.m_max(0),aabb.m_min(1),aabb.m_min(2));

	lines[18].set(aabb.m_min(0),aabb.m_max(1),aabb.m_max(2));
	lines[19].set(aabb.m_min(0),aabb.m_min(1),aabb.m_max(2));

	lines[20].set(aabb.m_min(0),aabb.m_min(1),aabb.m_max(2));
	lines[21].set(aabb.m_max(0),aabb.m_min(1),aabb.m_max(2));

	lines[22].set(aabb.m_max(0),aabb.m_min(1),aabb.m_max(2));
	lines[23].set(aabb.m_max(0),aabb.m_min(1),aabb.m_min(2));

	for (int i=0; i<24; i+=2)
	{
		displayLine(lines[i], lines[i+1], color, id, tag);
	}
}

void hkDebugDisplay::displayAabb(const hkTransform& transform, const hkAabb& aabb, hkColor::Argb color, int id, int tag)
{
	hkVector4	vertices[8];
	for(int i=0; i<8; ++i)
	{
		hkVector4	localVertex;
		localVertex.set(i&1? aabb.m_max.getComponent<0>() : aabb.m_min.getComponent<0>(),
						i&2? aabb.m_max.getComponent<1>() : aabb.m_min.getComponent<1>(),
						i&4? aabb.m_max.getComponent<2>() : aabb.m_min.getComponent<2>(),
						hkSimdReal::getConstant<HK_QUADREAL_0>());
		vertices[i]._setTransformedPos(transform, localVertex);
	}

	displayLine(vertices[0], vertices[1], color, id, tag);
	displayLine(vertices[1], vertices[3], color, id, tag);
	displayLine(vertices[3], vertices[2], color, id, tag);
	displayLine(vertices[2], vertices[0], color, id, tag);

	displayLine(vertices[4], vertices[5], color, id, tag);
	displayLine(vertices[5], vertices[7], color, id, tag);
	displayLine(vertices[7], vertices[6], color, id, tag);
	displayLine(vertices[6], vertices[4], color, id, tag);

	displayLine(vertices[0], vertices[4], color, id, tag);
	displayLine(vertices[1], vertices[5], color, id, tag);
	displayLine(vertices[2], vertices[6], color, id, tag);
	displayLine(vertices[3], vertices[7], color, id, tag);
}

void hkDebugDisplay::displayPlane(const hkVector4& plane, const hkVector4& offset, hkReal scale, hkColor::Argb color, int id, int tag)
{
	hkVector4 pos;

	pos.setAddMul(offset, plane, -plane.getComponent<3>());
	const hkVector4& normal = plane;

	int major = normal.getIndexOfMaxAbsComponent<3>();
	hkVector4 binorm; binorm.setZero();
	binorm((major+1) % 3) = 1;

	binorm.setCross(normal, binorm);
	binorm.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
	hkSimdReal s; s.setFromFloat(scale);
	binorm.mul(s);

	hkVector4 tangent;
	tangent.setCross(binorm, normal);
	tangent.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
	tangent.mul(s);

	//Draw the plane
	for (int e=0; e<4; e++)
	{
		hkVector4 pt1 = pos;
		(((e+0)%4)/2) ? pt1.sub(tangent) : pt1.add(tangent);
		(((e+1)%4)/2) ? pt1.sub(binorm)  : pt1.add(binorm);

		hkVector4 pt2 = pos;
		(((e+1)%4)/2) ? pt2.sub(tangent) : pt2.add(tangent);
		(((e+2)%4)/2) ? pt2.sub(binorm)  : pt2.add(binorm);	

		displayLine(pt1, pt2, color, id, tag);
	}

	//Draw the normal
	hkVector4 scaledNormal; scaledNormal.setMul(s, normal);
	displayArrow(pos, scaledNormal, color, id, tag);
}

void hkDebugDisplay::displayModelSpacePose(int numTransforms, const hkInt16* parentIndices, const hkQsTransform* modelSpacePose, const hkQsTransform& worldFromModel, int id, int tag, hkColor::Argb color)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{		
		m_debugDisplayHandlers[i]->displayModelSpacePose(numTransforms, parentIndices, modelSpacePose, worldFromModel, color, id, tag);	
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::displayLocalSpacePose(int numTransforms, const hkInt16* parentIndices, const hkQsTransform* localSpacePose, const hkQsTransform& worldFromModel, int id, int tag, hkColor::Argb color)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{		
		m_debugDisplayHandlers[i]->displayLocalSpacePose(numTransforms, parentIndices, localSpacePose, worldFromModel, color, id, tag);	
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::displayText(const char* text, hkColor::Argb color, int id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->displayText(text, color, id, tag);
	}
	m_arrayLock->leave();

}

void hkDebugDisplay::display3dText(const char* text, const hkVector4& pos, hkColor::Argb color, int id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->display3dText(text, pos, color, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::displayGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, const hkTransform& transform, hkColor::Argb color, int id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->displayGeometry(geometries, transform, color, id, tag);
	}
	m_arrayLock->leave();

}

void hkDebugDisplay::displayGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, hkColor::Argb color, int id, int tag)
{
	m_arrayLock->enter();
	for(int i = 0; i < m_debugDisplayHandlers.getSize(); i++)
	{
		m_debugDisplayHandlers[i]->displayGeometry(geometries, color, id, tag);
	}
	m_arrayLock->leave();
}

void hkDebugDisplay::displayLitTriangle(const hkVector4& a, const hkVector4& b, const hkVector4& c, hkColor::Argb color)
{
	hkVector4	ab; ab.setSub(b,a);
	hkVector4	ac; ac.setSub(c,a);
	hkVector4	n; n.setCross(ab,ac);
	if(n.normalizeIfNotZero<3>())
	{
		hkSimdReal zero9; zero9.setFromFloat(0.9f);
		hkVector4	light0; light0.set(0.5f,2,1); light0.normalize<3>(); light0.mul(zero9);
		hkVector4	light1; light1.set(-3,-2,-1); light1.normalize<3>(); light1.mul(zero9);
		hkVector4	colorBase; colorBase.set(hkColor::getRedAsFloat(color),hkColor::getGreenAsFloat(color),hkColor::getBlueAsFloat(color));
		hkVector4	color0; color0.set(0.75f,0.75f,0.75f,0); color0.mul(colorBase);
		hkVector4	color1; color1.set(0,0.05f,0.08f,0); color1.mul(colorBase);
		hkVector4	one = hkVector4::getConstant<HK_QUADREAL_1>();
		hkSimdReal	half = hkSimdReal::getConstant<HK_QUADREAL_INV_2>();
		hkVector4	finalColor;
		finalColor.setMul(light0.dot<3>(n)*half+half,color0);
		finalColor.addMul(light1.dot<3>(n)*half+half,color1);
		hkSimdReal rps; rps.setFromFloat(0.05f);
		finalColor.addMul(rps,colorBase);
		finalColor.setMax(finalColor,hkVector4::getZero());
		finalColor.setMin(finalColor,one);
		color = hkColor::rgbFromFloats(finalColor(0),finalColor(1),finalColor(2),hkColor::getAlphaAsFloat(color));
		HK_DISPLAY_TRIANGLE(a,b,c,color);
	}	
}

#if defined(HK_COMPILER_MWERKS)
#	pragma force_active on
#endif

HK_SINGLETON_IMPLEMENTATION(hkDebugDisplay);

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
