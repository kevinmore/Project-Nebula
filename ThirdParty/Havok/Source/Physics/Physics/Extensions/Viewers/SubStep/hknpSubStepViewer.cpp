/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/SubStep/hknpSubStepViewer.h>

#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>
#include <Physics/Physics/Extensions/Viewers/Shape/hknpShapeViewer.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplayGeometry.h>


int hknpSubStepViewer::s_tag = 0;
int hknpSubStepViewer::BodyTransforms::m_idOffsets[hknpSubStepViewer::BodyTransforms::MAX_NUM_TRANSFORMS] = { 0x010000, 0x020000, 0x040000, 0x080000 };

void HK_CALL hknpSubStepViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpSubStepViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpSubStepViewer( contexts );
}


hknpSubStepViewer::hknpSubStepViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts )
{
}

void hknpSubStepViewer::init()
{
	if ( m_context )
	{
		// Add any worlds that exist in the context
		for ( int i=0; i < m_context->getNumWorlds(); ++i )
		{
			worldAddedCallback( m_context->getWorld(i) );

			// Remember the world (could be one of many!)
			m_world = m_context->getWorld(i);
		}
	}
}

hknpSubStepViewer::~hknpSubStepViewer()
{
	removeAllBodies();

	if ( m_context )
	{
		for ( int i=0; i < m_context->getNumWorlds(); ++i )
		{
			worldRemovedCallback( m_context->getWorld(i) );
		}
	}
}

void hknpSubStepViewer::worldAddedCallback( hknpWorld* world )
{
	world->m_modifierManager->addModifier( hknpBody::IS_DYNAMIC, this );
}

void hknpSubStepViewer::worldRemovedCallback( hknpWorld* world )
{
	world->m_modifierManager->removeModifier( this );

	for ( int i=0; i<m_displayGeometries.getSize(); i++ )
	{
		m_displayGeometries[i]->removeReference();
	}
	m_displayGeometries.clear();
}


void hknpSubStepViewer::addBody( hknpBodyId bodyId, hkColor::Argb color )
{
	hkGeometry geometry;	// this needs to be declared here so the memory is still around till later

	const hknpBody& body = m_world->getBody(bodyId);

	if (body.isStatic())
	{
		return;
	}

	hknpShapeUtil::buildShapeDisplayGeometries(
		body.m_shape, hkTransform::getIdentity(), hkVector4::getConstant(HK_QUADREAL_1), hknpShape::CONVEX_RADIUS_DISPLAY_ROUNDED,
		m_displayGeometries );

	m_displayHandler->addGeometry( m_displayGeometries, body.getTransform(), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[0], s_tag, (hkUlong)-1 );
	m_displayHandler->addGeometry( m_displayGeometries, body.getTransform(), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[1], s_tag, (hkUlong)-1 );
	m_displayHandler->addGeometry( m_displayGeometries, body.getTransform(), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[2], s_tag, (hkUlong)-1 );
	m_displayHandler->addGeometry( m_displayGeometries, body.getTransform(), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[3], s_tag, (hkUlong)-1 );

	m_displayHandler->setGeometryColor( hkColor::replaceAlpha( 0x1f, color ), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[0], s_tag);
	m_displayHandler->setGeometryColor( hkColor::replaceAlpha( 0x3f, color ), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[1], s_tag);
	m_displayHandler->setGeometryColor( hkColor::replaceAlpha( 0x6f, color ), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[2], s_tag);
	m_displayHandler->setGeometryColor( hkColor::replaceAlpha( 0xaf, color ), (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[3], s_tag);

	hkTransform away; away.setIdentity(); away.getTranslation()(0) = 100000.0f;
	m_displayHandler->updateGeometry( away, (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[0], s_tag );
	m_displayHandler->updateGeometry( away, (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[1], s_tag );
	m_displayHandler->updateGeometry( away, (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[2], s_tag );
	m_displayHandler->updateGeometry( away, (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[3], s_tag );

	m_dynamicBodies.pushBack( bodyId );
	BodyTransforms& t = m_dynamicBodyTransforms.expandOne();
	t.m_nextTransform = 0;
}


void hknpSubStepViewer::removeBody( hknpBodyId bodyId )
{
	const hknpBody& body = m_world->getBody(bodyId);
	if ( body.isStatic() )
	{
		return;
	}

	int index = m_dynamicBodies.indexOf( bodyId );
	if ( index >= 0)
	{
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[0], s_tag, (hkUlong)-1);
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[1], s_tag, (hkUlong)-1);
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[2], s_tag, (hkUlong)-1);
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[3], s_tag, (hkUlong)-1);
		m_dynamicBodies.removeAt( index );
		m_dynamicBodyTransforms.removeAt( index );
	}
}


void hknpSubStepViewer::removeAllBodies()
{
	for (int i = 0; i < m_dynamicBodies.getSize(); ++i)
	{
		hknpBodyId bodyId = m_dynamicBodies[i];
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[0], s_tag, (hkUlong)-1);
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[1], s_tag, (hkUlong)-1);
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[2], s_tag, (hkUlong)-1);
		m_displayHandler->removeGeometry( (hkUlong)bodyId.value() + BodyTransforms::m_idOffsets[3], s_tag, (hkUlong)-1);
	}

	m_dynamicBodies.clear();
	m_dynamicBodyTransforms.clear();
}


void hknpSubStepViewer::step( hkReal deltaTime )
{
	HK_TIMER_BEGIN( "SubStepViewer", this );

	hkTransform away; away.setIdentity(); away.getTranslation()(0) = 100000.0f;
	for (int i = 0; i < m_dynamicBodies.getSize(); ++i)
	{
		hknpBodyId id = m_dynamicBodies[i];
		BodyTransforms& t = m_dynamicBodyTransforms[i];
		int ti = 0;
		for (; ti < t.m_nextTransform; ti++)
		{
			// Display colors from the end of the list when there's <4 bodies.
			m_displayHandler->updateGeometry( t.m_transforms[ti], (hkUlong)id.value() + BodyTransforms::m_idOffsets[4-t.m_nextTransform+ti], s_tag );
		}
		for (; ti < BodyTransforms::MAX_NUM_TRANSFORMS; ti++)
		{
			// Display colors from the end of the list when there's <4 bodies.
			m_displayHandler->updateGeometry( away, (hkUlong)id.value() + BodyTransforms::m_idOffsets[(4-t.m_nextTransform+ti)%4], s_tag );
		}
		t.m_nextTransform = 0;
	}

	HK_TIMER_END();
}


int hknpSubStepViewer::getEnabledFunctions()
{
	return (1<<FUNCTION_MANIFOLD_PROCESS);
}

void hknpSubStepViewer::manifoldProcessCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	hknpManifold* HK_RESTRICT manifold
	)
{
	// We could use a map from bodyId to array index.

	// get new transform & compare it with the old one, if it's exactly the same, than don't add it.
	// This should work, otherwise do some tolerance.

	const hknpBody* bodies[2] = { cdBodyA.m_body, cdBodyB.m_body };
	const hknpBodyId bodyIds[2] = { cdBodyA.m_body->m_id, cdBodyB.m_body->m_id };

	for (int bi = 0; bi < 2; bi++)
	{
		const hknpBody* body = bodies[bi];
		int bodyIdx = m_dynamicBodies.indexOf(bodyIds[bi]);
		if (-1 != bodyIdx)
		{
			BodyTransforms& t = m_dynamicBodyTransforms[bodyIdx];
			if (t.m_nextTransform == 0 || (t.m_nextTransform < 4 && !t.m_transforms[t.m_nextTransform].isApproximatelyEqual(body->getTransform())))
			{
				t.m_transforms[t.m_nextTransform++] = body->getTransform();
			}
		}
	}

	// For both bodies. update their position.

	// How do we know it's a next step ?

	//int id = manifold->m_bodyA;
	//const hkTransform& trA = bodyA->m_transformInOut;
	//const hkTransform& trB = bodyB->m_transformInOut;

	//hkVector4* nearestOnAinA = HK_NULL;
	//hkVector4* nearestOnBinB = HK_NULL;

	//hkVector4 planeBinWs = trB.getTranslation();
	//// 	planeBinWs.setRotatedDir(trA.getRotation(), planeBinA);
	//// 	planeBinWs.setXYZW( planeBinWs, planeBinA );

	//hkSimdReal pointScale(0.03f);

	//if(manifold->m_numPoints>0)
	//{
	//	for(int i=0;i<manifold->m_numPoints;++i)
	//	{
	//		hkVector4	x = manifold->m_positions[i];
	//		hkSimdReal distance = manifold->getDistance(i);
	//		manifold->m_positions[i](3) = distance.getReal();	// stick the distance in so that the convex hull can retrieve it in this debug code
	//		hkVector4	d; d.setMul( distance, manifold->m_normal);
	//		displayOrientedPoint(m_displayHandler, x, trA.getRotation(), pointScale, hkColor::LIGHTYELLOW, id, s_tag);
	//		x.add(d);
	//		displayOrientedPoint(m_displayHandler, x, trB.getRotation(), pointScale, hkColor::LIGHTCORAL, id, s_tag);

	//		//d.setMul(manifold.m_collisionCache->m_distanceOffset.getSimdAt(i)*hkSimdReal(100),normalWorld);
	//		//m_displayHandler->displayLine(x,d,hkColor::YELLOW, id, s_tag);
	//	}
	//	if(manifold->m_numPoints!=4 && nearestOnAinA)
	//	{
	//		hkVector4	worldA; worldA._setTransformedPos(trA,*nearestOnAinA);
	//		hkVector4	worldB; worldB._setTransformedPos(trB,*nearestOnBinB);
	//		hkVector4   cross;  cross.setCross(worldB, manifold->m_normal); cross.fastNormalize3();
	//		hkSimdReal dirlen = worldA.distanceTo(worldB);
	//		if (hkMath::isGreater(dirlen, hkSimdReal::getConstant(HK_QUADREAL_EPS)))
	//		{
	//			displayArrow(m_displayHandler, worldB, manifold->m_normal, cross, hkColor::WHITE, dirlen, id, s_tag);
	//			hkVector4	midPoint; midPoint.setInterpolate(worldA, worldB, hkSimdReal::getConstant(HK_QUADREAL_INV_2));
	//			hkStringBuf	s; s.printf("%d", (int)manifold->m_numPoints);
	//			m_displayHandler->display3dText(s.cString(),midPoint,hkColor::BLUE,id,s_tag);
	//		}
	//	}
	//	{
	//		hkgpConvexHull		hull;
	//		hull.buildPlanar(hkStridedVertices(manifold->m_positions,manifold->m_numPoints), planeBinWs);
	//		hkArray<hkVector4>	vertices;
	//		hull.fetchPositions(hkgpConvexHull::SOURCE_VERTICES,vertices);
	//		if(vertices.getSize())
	//		{
	//			hkVector4	cwa; cwa.setZero();
	//			hkVector4	cwb; cwb.setZero();
	//			for(int i=vertices.getSize()-1,j=0;j<vertices.getSize();i=j++)
	//			{
	//				hkVector4	vi=vertices[i],vj=vertices[j];
	//				m_displayHandler->displayLine(vi,vj,hkColor::LIGHTYELLOW, id, s_tag);
	//				cwa.add(vi);

	//				vi.addMul(vi.getSimdAt(3), manifold->m_normal);
	//				vj.addMul(vj.getSimdAt(3), manifold->m_normal);
	//				m_displayHandler->displayLine(vi,vj,hkColor::LIGHTCORAL, id, s_tag);
	//				cwb.add(vi);
	//			}
	//			hkSimdReal	f = hkSimdReal(1.0f/vertices.getSize());
	//			cwa.mul(f);
	//			cwb.mul(f);
	//			m_displayHandler->displayLine(cwa,cwb,hkColor::LIGHTSEAGREEN, id, s_tag);
	//		}
	//	}
	//}
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
