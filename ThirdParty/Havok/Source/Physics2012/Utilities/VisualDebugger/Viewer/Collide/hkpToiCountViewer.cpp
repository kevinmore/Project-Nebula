/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpToiCountViewer.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#define HK_TOI_COUNT_VIEWER_COLOR hkColor::MAROON
#define HK_TOI_COUNT_VIEWER_BITS_FOR_THIS_FRAME 8

// Defined in terms of the number of bits for the "this frame" part.
#define HK_TOI_COUNT_VIEWER_BITS_FOR_TOTAL ( 32 - HK_TOI_COUNT_VIEWER_BITS_FOR_THIS_FRAME )
#define HK_TOI_COUNT_VIEWER_HIGH_MASK ( ( 1 << HK_TOI_COUNT_VIEWER_BITS_FOR_THIS_FRAME ) - 1 )
#define HK_TOI_COUNT_VIEWER_LOW_MASK ( ( 1 << HK_TOI_COUNT_VIEWER_BITS_FOR_TOTAL ) - 1 )

int hkpToiCountViewer::s_tag = 0;

hkpToiCountViewer::hkpToiCountViewer( const hkArray<hkProcessContext*>& contexts )
: hkpWorldViewerBase( contexts)
{
}

void HK_CALL hkpToiCountViewer::registerViewer()
{
	s_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkProcess* HK_CALL hkpToiCountViewer::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpToiCountViewer(contexts);
}

int hkpToiCountViewer::getProcessTag()
{
	return s_tag;
}

void hkpToiCountViewer::entityRemovedCallback( hkpEntity* entity )
{
	entity->removeEntityListener( this );
	m_toiCounts.remove( static_cast<hkpRigidBody*>( entity ) );
}

void hkpToiCountViewer::entityDeletedCallback( hkpEntity* entity )
{
	entity->removeEntityListener( this );
	m_toiCounts.remove( static_cast<hkpRigidBody*>( entity ) );
}

void hkpToiCountViewer::init()
{
	for (int i=0; m_context && i < m_context->getNumWorlds(); ++i)
	{
		worldAddedCallback( m_context->getWorld(i));
	}
}

hkpToiCountViewer::~hkpToiCountViewer()
{
	for (int i=0; m_context && i < m_context->getNumWorlds(); ++i)
	{
		worldRemovedCallback( m_context->getWorld(i));
	}
}

void hkpToiCountViewer::worldAddedCallback( hkpWorld* w)
{
	w->markForWrite();
	w->addWorldPostSimulationListener( this );
	w->addContactListener( this );
	w->unmarkForWrite();
}

void hkpToiCountViewer::worldRemovedCallback( hkpWorld* w)
{
	w->markForWrite();
	// Remove this as an entity listener from any entities in the world, and remove
	// any such entities from the map.
	{
		hkArray<hkpRigidBody*> bodiesToRemove;
		// Iterate over all elements in the map, looking for ones in this world.
		for ( MapType::Iterator iterator = m_toiCounts.getIterator(); m_toiCounts.isValid( iterator ); iterator = m_toiCounts.getNext( iterator ) )
		{
			hkpRigidBody* body = m_toiCounts.getKey( iterator );
			if ( body->getWorld() == w )
			{
				bodiesToRemove.pushBack( body );
			}
		}
		const int numKeysToRemove = bodiesToRemove.getSize();
		for ( int i = 0; i < numKeysToRemove; ++i )
		{
			hkpRigidBody*& body = bodiesToRemove[i];
			body->removeEntityListener( this );
			m_toiCounts.remove( body );
		}	
	}
	w->removeContactListener( this );
	w->removeWorldPostSimulationListener( this );
	w->unmarkForWrite();
}

void hkpToiCountViewer::reset()
{
	// Remove this object as an entity listener from all bodies in the table.
	for ( MapType::Iterator iterator = m_toiCounts.getIterator(); m_toiCounts.isValid( iterator ); iterator = m_toiCounts.getNext( iterator ) )
	{
		hkpRigidBody* body = m_toiCounts.getKey( iterator );
		body->removeEntityListener( this );
	}
	// Clear the map.
	m_toiCounts.clear();
}

void hkpToiCountViewer::contactPointCallback( const hkpContactPointEvent& event )
{
	if ( event.isToi() )
	{
		for ( int i = 0; i < 2; ++i )
		{
			hkpRigidBody* body = event.getBody( i );
			MapType::Iterator iterator = m_toiCounts.findKey( body );
			// Is this body already in the map?
			if ( m_toiCounts.isValid( iterator ) )
			{
				const hkUint32 oldCount = m_toiCounts.getValue( iterator );
				const hkUint32 newCount = ( hkMath::min2( ( oldCount >> HK_TOI_COUNT_VIEWER_BITS_FOR_TOTAL ) + 1, HK_TOI_COUNT_VIEWER_HIGH_MASK ) << HK_TOI_COUNT_VIEWER_BITS_FOR_TOTAL ) 
					| ( hkMath::min2( ( oldCount & HK_TOI_COUNT_VIEWER_LOW_MASK ) + 1, HK_TOI_COUNT_VIEWER_LOW_MASK ) );
				// No locking required since contact point events for TOIs are fired single-threaded.
				m_toiCounts.setValue( iterator, newCount );
			}
			else
			{
				m_toiCounts.insert( body, 1 + ( 1 << HK_TOI_COUNT_VIEWER_BITS_FOR_TOTAL ) );
				// Listener for entity removed/deleted events so we can tidy up the map.
				body->addEntityListener( this );
			}
		}
	}
}

void hkpToiCountViewer::displayCountForBody( hkpRigidBody* body, DisplayPosition displayPosition )
{
	MapType::Iterator iterator = m_toiCounts.findKey( body );
	if ( m_toiCounts.isValid( iterator ) )
	{
		const hkUint32 count = m_toiCounts.getValue( iterator );
		const hkUint32 countThisFrame = count >> HK_TOI_COUNT_VIEWER_BITS_FOR_TOTAL;
		hkStringBuf s;
		if ( countThisFrame )
		{
			s.printf( "%d(%d)", count & HK_TOI_COUNT_VIEWER_LOW_MASK, countThisFrame );
		}
		else
		{
			s.printf( "%d", count & HK_TOI_COUNT_VIEWER_LOW_MASK );
		}
		
		displayTextAtBody( m_displayHandler, getProcessTag(), body, displayPosition, s.cString(), HK_TOI_COUNT_VIEWER_COLOR );
		// zero the count for this frame.
		m_toiCounts.setValue( iterator, count & HK_TOI_COUNT_VIEWER_LOW_MASK );
	}
}

void hkpToiCountViewer::postSimulationCallback( hkpWorld* world )
{
	// For each body in the world, check if we've recorded a TOI.
	const hkArray<hkpSimulationIsland*>& activeIslands = world->getActiveSimulationIslands();

	DisplayPosition displayPosition = getDisplayPositionFromGravityVector( world->getGravity() );

	// Active islands.
	for(int i = 0; i < activeIslands.getSize(); i++)
	{
		const hkArray<hkpEntity*>& activeEntities = activeIslands[i]->getEntities();
		for(int j = 0; j < activeEntities.getSize(); j++)
		{
			hkpRigidBody* body = static_cast<hkpRigidBody*>( activeEntities[j] );
			displayCountForBody( body, displayPosition );
		}
	}

	// The fixed island.
	{
		const hkArray<hkpEntity*>& fixedEntities = world->getFixedIsland()->getEntities();
		for(int j = 0; j < fixedEntities.getSize(); j++)
		{
			hkpRigidBody* body = static_cast<hkpRigidBody*>(fixedEntities[j]);
			displayCountForBody( body, displayPosition );
		}
	}
}

hkpToiCountViewer::DisplayPosition HK_CALL hkpToiCountViewer::getDisplayPositionFromGravityVector( const hkVector4& gravity )
{
	// We use the world's gravity to determine where to put the counts.
	int displayPosition;
	{
		const int majorAxis = gravity.getIndexOfMaxAbsComponent<3>();
		// We want the display position "opposite" gravity.
		displayPosition = majorAxis + ( gravity.getComponent( majorAxis ).isGreaterZero() ? 3 : 0 );
	}
	return (DisplayPosition) displayPosition;
}

void hkpToiCountViewer::displayTextAtBody( hkDebugDisplayHandler* handler, int tag, hkpRigidBody* body, DisplayPosition pos, const char* text, hkColor::Argb color, hkReal extraDistance, hkBool drawLine )
{
	HK_ASSERT2( 0x244fea87, body->getCollidable()->getShape() != HK_NULL, "A body must have a shape to have text displayed." );
	// The body's AABB.
	hkAabb aabb;
	{
		body->getCollidable()->getShape()->getAabb( body->getTransform(), 0.0f, aabb );
	}
	// Find the center of the face at which we should put the text.
	hkVector4 textPosition;
	hkVector4 faceCenter;
	{
		hkVector4 center;
		{
			center.setAdd( aabb.m_max, aabb.m_min );
			center.mul( hkSimdReal_Inv2 );
		}
		hkVector4 maxDir;
		{
			maxDir.setSub( aabb.m_max, center );
		}
		hkVector4 axis;
		{
			hkVector4 a; a.setAll(hkReal(1) - hkReal(2) * hkReal(pos / 3));
			axis.setMul( hkVector4::getConstant((hkVectorConstant)(HK_QUADREAL_1000 + (pos % 3))), a);
		}
		hkVector4 faceOffset;
		{
			hkSimdReal absDot; absDot.setAbs(axis.dot<3>( maxDir ));
			faceOffset.setMul( absDot, axis );
		}
		faceCenter.setAdd( faceOffset, center );
		textPosition.setAddMul( faceCenter, axis, hkSimdReal::fromFloat(extraDistance) );
	}

	handler->display3dText( text, textPosition, color, (int) (hkUlong) body->getCollidable(), tag );

	if ( drawLine )
	{
		const hkVector4& centerOfMass = body->getCenterOfMassInWorld();
		handler->displayLine( faceCenter, centerOfMass, color, (int) (hkUlong) body->getCollidable(), tag );
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
