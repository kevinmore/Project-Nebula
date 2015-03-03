/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpWeldingViewer.h>


int hkpWeldingViewer::m_tag = 0;


void HK_CALL hkpWeldingViewer::registerViewer( void )
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkProcess* HK_CALL hkpWeldingViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hkpWeldingViewer( contexts );
}

hkpWeldingViewer::hkpWeldingViewer(const hkArray<hkProcessContext*>& contexts)
: hkpWorldViewerBase( contexts )
{
	if ( m_context )
	{
		for ( int i = 0; i < m_context->getNumWorlds(); ++i )
		{
			hkpWorld* world = m_context->getWorld( i );

			world->markForWrite();
			world->addWorldPostSimulationListener( this );
			world->unmarkForWrite();
		}
	}
}

hkpWeldingViewer::~hkpWeldingViewer()
{
	if ( m_context )
	{
		for ( int i = 0; i < m_context->getNumWorlds(); ++i )
		{
			hkpWorld* world = m_context->getWorld( i );

			world->markForWrite();
			world->removeWorldPostSimulationListener( this );
			world->unmarkForWrite();
		}
	}
}

void hkpWeldingViewer::worldAddedCallback( hkpWorld* world )
{
	world->markForWrite();;
	world->addWorldPostSimulationListener(this);
	world->unmarkForWrite();
}

void hkpWeldingViewer::worldRemovedCallback( hkpWorld* world )
{
	world->markForWrite();;
	world->removeWorldPostSimulationListener(this);
	world->unmarkForWrite();
}


void hkpWeldingViewer::postSimulationCallback( hkpWorld* world )
{
	HK_TIME_CODE_BLOCK("hkpWeldingViewer::postSimulationCallback", HK_NULL);

	world->lock();

	hkpPhysicsSystem* system = world->getWorldAsOneSystem();
	const hkArray< hkpRigidBody* >& rigidBodies = system->getRigidBodies();

	// Iterate over all rigid bodies
	for ( int i = 0; i < rigidBodies.getSize(); ++i )
	{
		// Read body data
		const hkpRigidBody* body = rigidBodies[ i ];
		const int collidableId = (int)(hkUlong)(body->getCollidable());
		const hkTransform& bodyTransform = body->getTransform();
		const hkpShape* rootShape = body->getCollidable()->getShape();
		
		// Check if the shape is a shape container and skip if it is not
		const hkpShapeContainer* shapeContainer = HK_NULL;
		shapeContainer = rootShape->getContainer();				
		if (!shapeContainer)
		{
			continue;
		}

		// Iterate over all contained shapes
		for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey( key ) )
		{
			hkpShapeBuffer buffer;
			const hkpShape* childShape = shapeContainer->getChildShape( key, buffer );

			// Skip all but triangles
			if ( childShape->getType() !=  hkcdShapeType::TRIANGLE )
			{
				continue;
			}
			const hkpTriangleShape* triangle = static_cast< const hkpTriangleShape* >( childShape );

			// Do not show two sided welding
			hkpWeldingUtility::WeldingType weldingType = triangle->getWeldingType();
			if (weldingType == hkpWeldingUtility::WELDING_TYPE_TWO_SIDED || weldingType == hkpWeldingUtility::WELDING_TYPE_NONE)
			{
				continue;
			}

			// Transform triangle
			hkVector4 vertices[ 3 ];
			vertices[ 0 ]._setTransformedPos( bodyTransform, triangle->getVertex<0>() );
			vertices[ 1 ]._setTransformedPos( bodyTransform, triangle->getVertex<1>() );
			vertices[ 2 ]._setTransformedPos( bodyTransform, triangle->getVertex<2>() );

			// Calculate triangle center and normal
			hkVector4 triangleCenter; 
			triangleCenter.setAdd( vertices[0], vertices[1] );
			triangleCenter.add( vertices[2] );
			triangleCenter.mul( hkSimdReal_Inv3 );
			hkVector4 triangleNormal;
			hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal( vertices[ 0 ], vertices[ 1 ], vertices[ 2 ], triangleNormal );

			// Draw the snap vectors for each edge 
			hkUint16 weldingInfo = triangle->getWeldingInfo();
			for ( int k = 0; k < 3; ++k )
			{
				int edgeIndex0 = k;
				int edgeIndex1 = ( k + 1 ) % 3;

				// Calculate normalized edge direction
				hkVector4 edge;
				edge.setSub( vertices[ edgeIndex1 ], vertices[ edgeIndex0 ] );
				edge.normalize<3>();

				// Calculate slightly shifted edge center
				hkVector4 edgeCenter;
				edgeCenter.setInterpolate( vertices[ edgeIndex0 ], vertices[ edgeIndex1 ], hkSimdReal::fromFloat(0.48f) );

				// Obtain snap vector
				hkVector4 snapVector0;
				int edgeCode = hkpMeshWeldingUtility::calcSingleEdgeBitcode( weldingInfo, k );
				hkpWeldingUtility::calcSnapVectorOneSided( triangleNormal, edge, edgeCode, triangle->getWeldingType(), snapVector0 );

				// Invert triangle normal in clockwise welding
				if ( triangle->getWeldingType() == hkpWeldingUtility::WELDING_TYPE_CLOCKWISE )
				{
					triangleNormal.setNeg<3>( triangleNormal );
				}

				// Discard snap vectors too opposed to triangle normal
				if ( snapVector0.dot<3>( triangleNormal ) < hkSimdReal::fromFloat(-0.98f) )
				{
					continue;
				}

				// Draw snap vector, triangle normal and a line from edge center to triangle center
				hkVector4 p0, p1;
				p0.setAdd( edgeCenter, snapVector0 );
				p1.setAdd( edgeCenter, triangleNormal );
				//m_displayHandler->displayArrow( edgeCenter, snapVector0, hkColor::YELLOW, collidableId, m_tag );
				//m_displayHandler->displayArrow( edgeCenter, triangleNormal, hkColor::YELLOW, collidableId, m_tag );
				m_displayHandler->displayLine( edgeCenter, p0, hkColor::ORANGE, collidableId, m_tag );
				m_displayHandler->displayLine( edgeCenter, p1, hkColor::ORANGE, collidableId, m_tag );
				m_displayHandler->displayLine( edgeCenter, triangleCenter, hkColor::GREEN, collidableId, m_tag );

				// Draw an arc between snap vector and triangle normal
				hkVector4 start = snapVector0;
				const int kSteps = 10;
				hkSimdReal invKSteps; invKSteps.setReciprocal(hkSimdReal::fromInt32(kSteps));
				for ( int s = 0; s < kSteps; ++s )
				{
					hkVector4 interp;
					interp.setInterpolate( snapVector0, triangleNormal, hkSimdReal::fromInt32(s + 1) * invKSteps  );
					interp.normalize<3>();

					hkVector4 ep0, ep1;
					ep0.setAdd( edgeCenter, start );
					ep1.setAdd( edgeCenter, interp );

					m_displayHandler->displayLine( ep0, ep1, hkColor::ORANGE, collidableId, m_tag );

					start = interp;
				}
			}							
		}
	}

	system->removeReference();
	world->unlock();
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
