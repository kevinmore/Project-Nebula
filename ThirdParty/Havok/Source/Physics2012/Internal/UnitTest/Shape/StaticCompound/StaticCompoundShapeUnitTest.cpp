/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Internal/hkpInternal.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>

static hkpStaticCompoundShape* createStaticCompoundShape( hkArray<hkpStaticCompoundShape*>& scsChildren, bool includeScsChildren = true )
{
	hkpStaticCompoundShape* compoundShape = new hkpStaticCompoundShape();

	hkpSphereShape* sphere = new hkpSphereShape( 1.0f );
	for( int i=0; i<10; ++i )
	{
		int mod = includeScsChildren ? 3 : 2;
		switch( i % mod )
		{
		case 0:
			{
				compoundShape->addInstance( sphere, hkQsTransform::getIdentity() );
				break;
			}
		case 1:
			{
				hkArray<hkpShape*> shapes(i*5, sphere);
				hkpListShape* list = new hkpListShape( shapes.begin(), shapes.getSize() );
				compoundShape->addInstance( list, hkQsTransform::getIdentity() );
				list->removeReference();
				break;
			}
		case 2:
			{
				hkpShape* scs = createStaticCompoundShape( scsChildren, false );
				scsChildren.pushBack( (hkpStaticCompoundShape*)scs );
				compoundShape->addInstance( scs, hkQsTransform::getIdentity() );
				scs->removeReference();
				break;
			}
		default:
			break;
		}
	}

	sphere->removeReference();
	compoundShape->bake();
	return compoundShape;
}

// Check that disabling keys and instances works as expected
int hkpStaticCompoundShape_disableKeys_test()
{
	//
	// Create a compound shape with a mix of convex children and collections of varying size
	//

	hkArray<hkpStaticCompoundShape*> scsChildren;
	hkpStaticCompoundShape* compoundShape = createStaticCompoundShape( scsChildren );

	//
	// Check shape key disabling/enabling
	//

	// Get a large AABB covering the whole shape
	hkAabb aabb;
	compoundShape->getAabb( hkTransform::getIdentity(), 100.0f, aabb );

	// Check that AABB queries give correct number of hits
	hkArray<hkpShapeKey> keys;
	const int numChildren = compoundShape->getNumChildShapes();
	compoundShape->queryAabb( aabb, keys );
	HK_TEST( numChildren == keys.getSize() );
	HK_TEST( numChildren == (int)compoundShape->queryAabbImpl( aabb, keys.begin(), keys.getSize() ) );

	// Check disabling
	for( hkpShapeKey key = compoundShape->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = compoundShape->getNextKey(key) )
	{
		HK_TEST( compoundShape->isShapeKeyEnabled(key) );
		compoundShape->setShapeKeyEnabled(key, false);
		HK_TEST( !compoundShape->isShapeKeyEnabled(key) );

		// Should be 1 fewer result in the AABB queries
		const int expectedNumKeys = keys.getSize() - 1;
		
		keys.clear();
		compoundShape->queryAabb( aabb, keys );
		HK_TEST( expectedNumKeys == keys.getSize() );
		
		keys.setSize( HK_MAX_NUM_HITS_PER_AABB_QUERY );
		keys.setSize( (int)compoundShape->queryAabbImpl( aabb, keys.begin(), keys.getSize() ) );
		HK_TEST( expectedNumKeys == keys.getSize() );

		
	}

	// Check enabling
	compoundShape->queryAabb( aabb, keys );
	for( hkpShapeKey key = compoundShape->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = compoundShape->getNextKey(key) )
	{
		HK_TEST( !compoundShape->isShapeKeyEnabled(key) );
		compoundShape->setShapeKeyEnabled(key, true);
		HK_TEST(  compoundShape->isShapeKeyEnabled(key) );

		// Should be 1 more result in the AABB queries
		const int expectedNumKeys = keys.getSize() + 1;
		
		keys.clear();
		compoundShape->queryAabb( aabb, keys );
		HK_TEST( expectedNumKeys == keys.getSize() );
		
		keys.setSize( HK_MAX_NUM_HITS_PER_AABB_QUERY );
		keys.setSize( (int)compoundShape->queryAabbImpl( aabb, keys.begin(), keys.getSize() ) );
		HK_TEST( expectedNumKeys == keys.getSize() );

		
	}

	//
	// Check instance disabling/enabling
	//

	// Check disabling
	for( int instanceId = 0; instanceId < compoundShape->getInstances().getSize(); instanceId++ )
	{
		HK_TEST( compoundShape->isInstanceEnabled(instanceId) );
		compoundShape->setInstanceEnabled(instanceId, false);
		HK_TEST( !compoundShape->isInstanceEnabled(instanceId) );

		// Also check that all child shape keys are reported as disabled
		const hkpStaticCompoundShape::Instance& instance = compoundShape->getInstances()[instanceId];
		const hkUint32 flags = instance.getFlags();
		if( flags & hkpStaticCompoundShape::Instance::FLAG_IS_LEAF )
		{
			hkpShapeKey key = compoundShape->composeShapeKey( instanceId, 0 );
			HK_TEST( !compoundShape->isShapeKeyEnabled(key) );
		}
		else
		{
			const hkpShapeContainer* container = instance.getShape()->getContainer();
			HK_TEST( container != HK_NULL );
			for( hkpShapeKey childKey = container->getFirstKey(); childKey != HK_INVALID_SHAPE_KEY; childKey = container->getNextKey(childKey) )
			{
				hkpShapeKey key = compoundShape->composeShapeKey( instanceId, childKey );
				HK_TEST( !compoundShape->isShapeKeyEnabled(key) );
			}
		}
	}

	// Check enabling
	for( int instanceId = 0; instanceId < compoundShape->getInstances().getSize(); instanceId++ )
	{
		HK_TEST( !compoundShape->isInstanceEnabled(instanceId) );
		compoundShape->setInstanceEnabled(instanceId, true);
		HK_TEST( compoundShape->isInstanceEnabled(instanceId) );

		// Also check that all child shape keys are reported as enabled.
		// NOTE: This isn't a proper test, as individual shape keys could still be disabled in real use cases.
		const hkpStaticCompoundShape::Instance& instance = compoundShape->getInstances()[instanceId];
		const hkUint32 flags = instance.getFlags();
		if( flags & hkpStaticCompoundShape::Instance::FLAG_IS_LEAF )
		{
			hkpShapeKey key = compoundShape->composeShapeKey( instanceId, 0 );
			HK_TEST( compoundShape->isShapeKeyEnabled(key) );
		}
		else
		{
			const hkpShapeContainer* container = instance.getShape()->getContainer();
			HK_TEST( container != HK_NULL );
			for( hkpShapeKey childKey = container->getFirstKey(); childKey != HK_INVALID_SHAPE_KEY; childKey = container->getNextKey(childKey) )
			{
				hkpShapeKey key = compoundShape->composeShapeKey( instanceId, childKey );
				HK_TEST( compoundShape->isShapeKeyEnabled(key) );
			}
		}
	}

	//
	// Check that disabling child scs is correctly reported in parent
	//

	HK_TEST( compoundShape->isShapeKeyEnabled( compoundShape->composeShapeKey(2,0) ) );
	scsChildren[0]->setInstanceEnabled( 0, false );
	HK_TEST( !compoundShape->isShapeKeyEnabled( compoundShape->composeShapeKey(2,0) ) );

	// Clean up
	compoundShape->removeReference();
	return 0;
}


//
// test registration
//
#if defined( HK_COMPILER_MWERKS )
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( hkpStaticCompoundShape_disableKeys_test , "Fast", "Physics2012/Test/UnitTest/Internal/", __FILE__ );

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
