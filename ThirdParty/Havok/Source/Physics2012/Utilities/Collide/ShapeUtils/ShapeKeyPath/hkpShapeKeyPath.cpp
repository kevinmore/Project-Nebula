/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeKeyPath/hkpShapeKeyPath.h>

#include <Physics2012/Collide/Shape/hkpShapeContainer.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastOutput.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>


//
// ShapeKeyPath Construction
//  

hkpShapeKeyPath::hkpShapeKeyPath( const hkpContactPointEvent& event, int bodyIdx ) 
{
	HK_ASSERT2( 0x4ae62bd4,bodyIdx == 0 || bodyIdx == 1  ,"Body index needs to be 0 or 1" );	
	init( event.getBody(bodyIdx)->getCollidable()->getShape(), event.getShapeKeys(bodyIdx), event.getBody(bodyIdx)->m_numShapeKeysInContactPointProperties );
	m_isOrderLeafToRoot = true;
}

hkpShapeKeyPath::hkpShapeKeyPath( const hkpWorldRayCastOutput& output ) 
{
	HK_ASSERT2( 0x28512f89, output.hasHit(), "No hit detected" );
	init( output.m_rootCollidable->getShape(), output.m_shapeKeys, hkpShapeRayCastOutput::MAX_HIERARCHY_DEPTH );
	m_isOrderLeafToRoot = false;
}

hkpShapeKeyPath::hkpShapeKeyPath( const hkpShape* shape, const hkpShapeRayCastOutput& output )
{
	HK_ASSERT2( 0x28512f90, output.hasHit(), "No hit detected" );
	init( shape, output.m_shapeKeys, hkpShapeRayCastOutput::MAX_HIERARCHY_DEPTH );
	m_isOrderLeafToRoot = false;
}

// Common initialization.
void hkpShapeKeyPath::init( const hkpShape* shape, const hkpShapeKey* keys, int maxKeys )
{
	m_rootShape		= shape;
	m_keys			= keys;
	m_numKeys		= 0;

    while ( ( m_numKeys < maxKeys ) && ( m_keys[m_numKeys] != HK_INVALID_SHAPE_KEY ) )
    {
        ++m_numKeys;
    }
}


//
// ShapeKeyPath utility functions.
// 

hkpShapeKey hkpShapeKeyPath::getShapeKey( int keyIndex ) const
{
    if( keyIndex < m_numKeys )
    {
		// The current shape is a container with several sub-shapes.
		if ( m_isOrderLeafToRoot )
        {
            // Shapes keys are stored in leaf-to-root order, but the last one is an invalid.
           return m_keys[m_numKeys - keyIndex - 1];
        }
        else
        {
            return m_keys[keyIndex];
        }
    }
    else
    {
        // The current shape was a leaf.
		return HK_INVALID_SHAPE_KEY;
    }
}

void hkpShapeKeyPath::getShapes( int maxShapesOut, hkpShapeBuffer* buffers, const hkpShape** shapesOut, int& numShapesOut)
{
	HK_ASSERT( 0x2e26bc2d, maxShapesOut > 0 );

	numShapesOut = 0;
	Iterator iterator = getIterator();

	// Traverse all shapes. 
	while( iterator.isValid() )
	{
		shapesOut[numShapesOut] = iterator.getShape();
		iterator.nextImpl( buffers + numShapesOut );
		++numShapesOut;
		HK_ASSERT2( 0x2e26bc2e, numShapesOut < maxShapesOut ,"Maximum number of shapes reached" );
	}
}


//
// Iterator functions.
// 

hkpShapeKeyPath::Iterator::Iterator( const hkpShapeKeyPath* path, const hkpShape* rootShape )
:	m_path(path), m_currentShape(rootShape), m_currentKeyIdx(0), m_isValid(true)
{
}

void hkpShapeKeyPath::Iterator::nextImpl( hkpShapeBuffer* buf ) 
{
    HK_ASSERT2( 0x5766e40f, isValid() ,"Invalid iterator" );
	hkpShapeKey nextKey = m_path->getShapeKey( m_currentKeyIdx );

	if ( nextKey == HK_INVALID_SHAPE_KEY)
	{
		if ( ( m_currentShape->getType() == hkcdShapeType::CONVEX_TRANSFORM ) || ( m_currentShape->getType() == hkcdShapeType::CONVEX_TRANSLATE ) )
		{
			// The subsequent shapes are just transforms or translates, which have only one children.
			nextKey = 0;
		}
		else
		{
			// The current shape was a leaf, so we invalidate the iterator and return immediately.
			m_isValid = false;
			m_currentShape = HK_NULL;
			return;
		}
	}

	m_currentShape = m_currentShape->getContainer()->getChildShape( nextKey, *buf );

	++m_currentKeyIdx;
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
