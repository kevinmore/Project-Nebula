/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/WeldingTriangle/hknpWeldingTriangleViewer.h>

#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


int hknpWeldingTriangleViewer::s_tag = 0;

void HK_CALL hknpWeldingTriangleViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpWeldingTriangleViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpWeldingTriangleViewer( contexts );
}


hknpWeldingTriangleViewer::hknpWeldingTriangleViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts )
{
	m_semifinishedTriangleShape = hknpTriangleShape::createEmptyTriangleShape( 0.0f );
}

hknpWeldingTriangleViewer::~hknpWeldingTriangleViewer()
{
	if( m_context )
	{
		for( int i=0; i < m_context->getNumWorlds(); ++i )
		{
			worldRemovedCallback( m_context->getWorld(i) );
		}
	}
	delete m_semifinishedTriangleShape;
}

void hknpWeldingTriangleViewer::worldAddedCallback( hknpWorld* world )
{
	world->m_modifierManager->incrementGlobalBodyFlags( hknpBody::RAISE_MANIFOLD_PROCESSED_EVENTS );
	world->getEventSignal( hknpEventType::MANIFOLD_PROCESSED ).subscribe( this, &hknpWeldingTriangleViewer::onManifoldProcessedEvent, "ManifoldViewer");
}

void hknpWeldingTriangleViewer::worldRemovedCallback( hknpWorld* world )
{
	world->m_modifierManager->decrementGlobalBodyFlags( hknpBody::RAISE_MANIFOLD_PROCESSED_EVENTS );
	world->getEventSignal( hknpEventType::MANIFOLD_PROCESSED ).unsubscribeAll( this );
}

static HK_FORCE_INLINE void drawTriangle(
	hkDebugDisplayHandler* displayHandler, hkVector4Parameter A, hkVector4Parameter B, hkVector4Parameter C,
	hkColor::Argb color, hkVector4Parameter offset)
{
	hkVector4 v0; v0.setAdd(A, offset);
	hkVector4 v1; v1.setAdd(B, offset);
	hkVector4 v2; v2.setAdd(C, offset);
	displayHandler->displayTriangle( v0, v1, v2, color, 0, hknpWeldingTriangleViewer::s_tag );
}


void hknpWeldingTriangleViewer::onManifoldProcessedEvent( const hknpEventHandlerInput& input, const hknpEvent& event )
{
	HK_TIME_CODE_BLOCK( "WeldingTriangleViewer", HK_NULL );

	const hknpManifoldProcessedEvent& cpsEvent = event.asManifoldProcessedEvent();

	const hknpBody& bodyA = input.m_world->getBody(cpsEvent.m_bodyIds[0]);
	const hknpBody& bodyB = input.m_world->getBody(cpsEvent.m_bodyIds[1]);
	const hknpShape* shapeB = bodyB.m_shape;

	// Only show info if shapeB is a composite and welding is enabled.
	{
		if( shapeB->m_dispatchType != hknpCollisionDispatchType::COMPOSITE )
		{
			return;
		}

		const hknpBodyQualityLibrary* qualityLibrary = input.m_simulationThreadContext->m_world->getBodyQualityLibrary();
		const hknpBodyQuality* qA = &qualityLibrary->getEntry(bodyA.m_qualityId);
		const hknpBodyQuality* qB = &qualityLibrary->getEntry(bodyB.m_qualityId);

		hknpBodyQuality::Flags qualityFlags;
		/*const hknpBodyQuality* q =*/ hknpBodyQuality::combineBodyQualities(qA, qB, &qualityFlags );
		if( !qualityFlags.anyIsSet(hknpBodyQuality::ANY_WELDING) )
		{
			return;
		}
	}

	// Query sub-shape to find the triangle (or quad)
	hkVector4 vertices[4];
	int numVertices;
	{
		hknpShapeCollector leafShapeCollector( m_semifinishedTriangleShape );
		leafShapeCollector.reset( bodyB.getTransform() );

		shapeB->getLeafShape( cpsEvent.m_shapeKeys[1], &leafShapeCollector );

		const hkVector4* tV = m_semifinishedTriangleShape->getVertices();
		numVertices = m_semifinishedTriangleShape->isQuad() ? 4 : 3;
		for (int i=0; i<numVertices; i++)
		{
			vertices[i].setTransformedPos( leafShapeCollector.m_transformOut, tV[i] );
		}
	}

	// Calculate triangle normal
	hkVector4 triangleNormal;
	hkVector4 triangleCenter;
	{
		triangleCenter.setAdd(vertices[0], vertices[1]);
		triangleCenter.add(vertices[2]);
		triangleCenter.mul(hkSimdReal_Inv3);

		hkVector4 eA; eA.setSub(vertices[1], vertices[0]);
		hkVector4 eB; eB.setSub(vertices[2], vertices[0]);
		triangleNormal.setCross(eA, eB);
		triangleNormal.normalize<3>();
	}

	// Draw triangle(s)
	{
		hkVector4 offset;
		offset.setMul( hkSimdReal::fromFloat(0.01f), triangleNormal );
		drawTriangle(m_displayHandler, vertices[0], vertices[1], vertices[2], hkColor::LIGHTSEAGREEN, offset);
		if( numVertices == 4 )
		{
			drawTriangle(m_displayHandler, vertices[0], vertices[2], vertices[3], hkColor::LIGHTSEAGREEN, offset);
		}
	}

	// Draw normals
	{
		hkVector4 triangleEnd; triangleEnd.setAddMul(triangleCenter, triangleNormal, hkSimdReal_Inv2);
		hkVector4 normalEnd; normalEnd.setAddMul(triangleCenter, cpsEvent.m_manifold.m_normal, hkSimdReal_Inv2);

		m_displayHandler->displayLine(triangleCenter, triangleEnd, hkColor::BLUE, 0, hknpWeldingTriangleViewer::s_tag);
		m_displayHandler->displayLine(triangleCenter, normalEnd, hkColor::YELLOW, 0, hknpWeldingTriangleViewer::s_tag);
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
