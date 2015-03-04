/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>

// Collection of utility functions to test hkpShape interface methods on a hkpShape object regardless of
// its dynamic type
namespace ShapeInterfaceTester
{
	// Performs getMaximumProjection in the 6 axis directions with different scales and checks that 
	// the results are consistent with the scaled aabbs
	static void testGetMaximumProjection(const hkpShape* shape)
	{
		hkAabb aabb; 
		shape->getAabb(hkTransform::getIdentity(), 0, aabb);
		hkVector4 maxProjectionDifference; aabb.getHalfExtents(maxProjectionDifference);
		maxProjectionDifference.mul(hkSimdReal::fromFloat(0.02f));

		const hkReal scales[] = { 2.0f, 1.0f, 0.5f };
		const hkReal epsilon = 0.00001f;

		for (int i = 0; i < 3; ++i)
		{
			const hkReal scale = scales[i];

			for (int j = 0; j < 6; ++j)
			{	
				// Calculate projection direction and obtain maximum projection
				int axis = j / 2;		
				hkReal sign = -1.0f + 2.0f * (j & 1);
				hkVector4 direction; direction.setZero();
				direction(axis) = sign * scale;
				hkReal projection = shape->getMaximumProjection(direction) / scale;

				// Create a string to help identify the test in case of failure
				hkStringBuf testDescription;
				testDescription.printf("shape type %d, direction (%.2f, %.2f, %.2f)", shape->getType(), direction(0), direction(1), direction(2));

				// Check that the projection is inside the aabb
				hkReal signedProjection = projection * sign;				
				HK_TEST2(aabb.m_min(axis) - signedProjection <= epsilon, testDescription);
				HK_TEST2(aabb.m_max(axis) - signedProjection >= -epsilon, testDescription);

				// Check that the difference between the projection and the corresponding aabb boundary is less 
				// than 1% of the aabb extent
				hkVector4 boundary; boundary.setSelect(direction.signBitSet(), aabb.m_min, aabb.m_max);
				HK_TEST2(hkMath::equal(boundary(axis), signedProjection, maxProjectionDifference(axis)), testDescription);
			}
		}
	}
}


namespace
{
	class TestSampledHeightFieldShape : public hkpSampledHeightFieldShape
	{
		public:

			TestSampledHeightFieldShape(const hkpSampledHeightFieldBaseCinfo& info) : hkpSampledHeightFieldShape(info) {}
			HK_FORCE_INLINE hkReal getHeightAtImpl(int x, int z) const { return hkReal(x + z); }
			HK_FORCE_INLINE hkBool getTriangleFlipImpl() const { return false; }
			virtual void collideSpheres(const CollideSpheresInput& input, SphereCollisionOutput* outputArray) const
			{
				return hkSampledHeightFieldShape_collideSpheres(*this, input, outputArray);
			}
	};
}


// Test all different implementations of getMaximumProjection
static void testGetMaximumProjection()
{
	hkArray< hkRefPtr<hkpShape> > shapes;

	// Default implementation (hkpShape)
	{
		hkpSampledHeightFieldBaseCinfo info;
		shapes.pushBack(hkRefNew<hkpShape>(new TestSampledHeightFieldShape(info)));
	}

	// hkpConvexShape implementation
	hkRefPtr<hkpConvexShape> convexShape;
	{	
		hkVector4 halfExtents; halfExtents.set(0.95f, 0.95f, 0.95f);
		const hkReal convexRadius = 0.05f;
		convexShape = hkRefNew<hkpConvexShape>(new hkpBoxShape(halfExtents, convexRadius));
		shapes.pushBack(convexShape.val());
	}

	// hkpConvexTranslateShape implementation
	{
		hkVector4 translation; translation.set(1, -2, 3);
		shapes.pushBack(hkRefNew<hkpShape>(new hkpConvexTranslateShape(convexShape, translation)));
	}

	// hkpShapeContainer implementation
	hkArray<hkpShape*> shapesInList;
	{
		hkVector4 translation; translation.set(1, -2, 3);
		shapesInList.pushBack(new hkpConvexTranslateShape(convexShape, translation));
		translation.set(-3, 2, -1);
		shapesInList.pushBack(new hkpConvexTranslateShape(convexShape, translation));

		shapes.pushBack(hkRefNew<hkpShape>(new hkpListShape(shapesInList.begin(), shapesInList.getSize(), 
			hkpShapeContainer::REFERENCE_POLICY_IGNORE)));
	}

	// hkpStaticCompoundShape implementation
	{
		hkpStaticCompoundShape* compound = new hkpStaticCompoundShape();		
		hkVector4 translation;
		hkQuaternion rotation;
		hkVector4 scale;

		{
			translation.set(1, -2, 3);
			rotation.setFromEulerAngles(0.5f * HK_REAL_PI, -0.5f * HK_REAL_PI, 0);
			scale.set(2, -1, 0.5f);
			hkQsTransform transform; transform.set(translation, rotation, scale);
			compound->addInstance(convexShape, transform);				
		}

		{
			translation.set(-3, 2, -1);
			rotation.setFromEulerAngles(0, 0.5f * HK_REAL_PI, -0.5f * HK_REAL_PI);
			scale.set(-0.5f, 1, -2);
			hkQsTransform transform; transform.set(translation, rotation, scale);
			compound->addInstance(convexShape, transform);				
		}

		compound->bake();
		shapes.pushBack(hkRefNew<hkpShape>(compound));		
	}

	// hkpTransformShape implementation
	{
		hkVector4 translation; translation.set(1, -2, 3);
		hkQuaternion rotation; rotation.setFromEulerAngles(0.5f * HK_REAL_PI, -0.5f * HK_REAL_PI, 0);
		hkTransform transform; transform.set(rotation, translation);
		shapes.pushBack(hkRefNew<hkpShape>(new hkpTransformShape(convexShape, transform)));
	}

	for (int i = 0; i < shapes.getSize(); ++i)
	{
		ShapeInterfaceTester::testGetMaximumProjection(shapes[i]);
	}
}


int ShapeInterfaceUnitTest()
{
	testGetMaximumProjection();

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
	#pragma fullpath_file on
#endif

HK_TEST_REGISTER( ShapeInterfaceUnitTest, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__ );

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
