/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Internal/hkcdInternal.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastSphere.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>
#include <Common/Base/System/Stopwatch/hkSystemClock.h>	
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
	


// Anonymous namespace used to force internal linkage for struct names
namespace
{
	// Test definition in terms of inputs and expected outputs
	struct TestCase
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DEMO, TestCase);

		// hkVector4 wrapper used to hack around the lack of constructors
		struct myVector
		{
			hkVector4 data;

			myVector(hkReal x, hkReal y, hkReal z, hkReal w = 0)
			{
				data.set(x, y, z, w);
			}
			myVector() {}
		};

		// Name
		hkStringPtr m_name;	

		// Test input
		myVector m_rayStart;
		myVector m_rayEnd;
		myVector m_spherePosAndRadius;
		hkcdRayQueryFlags::Enum m_queryFlags;

		// Expected outputs
		hkBool m_result;
		hkSimdReal m_hitFraction;
		myVector m_normal;	

		// Positive intersection test case
		TestCase(const char * name, const myVector& rayStart, const myVector& rayEnd, const myVector& spherePosAndRadius, 
			hkcdRayQueryFlags::Enum queryFlags,
			const hkSimdReal& hitFraction, const myVector& normal) :
		m_name(name), m_rayStart(rayStart), m_rayEnd(rayEnd), m_spherePosAndRadius(spherePosAndRadius), m_queryFlags(queryFlags),
			m_result(true), m_hitFraction(hitFraction), m_normal(normal) {}

		// Negative intersection test case
		TestCase(const char * name, const myVector& rayStart, const myVector& rayEnd, const myVector& spherePosAndRadius, hkcdRayQueryFlags::Enum queryFlags) :
		m_name(name), m_rayStart(rayStart), m_rayEnd(rayEnd), m_spherePosAndRadius(spherePosAndRadius), m_queryFlags(queryFlags), 
			m_result(false) {}
	};
}


// Anonymous namespace for a reference ray-sphere test using double precision.
namespace 
{
	// Returns the maximum value for this type.
	template <typename T>
	static HK_FORCE_INLINE T HK_CALL hkTypeMax();

	template <>
	HK_FORCE_INLINE hkDouble64 HK_CALL hkTypeMax()
	{
		return 1.7970e+308;
	}

	// A template based hkVector4 class with limited implementation.
	template <typename T>
	struct hkVectorT4
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DEMO, hkVectorT4<T>);


		HK_FORCE_INLINE hkVectorT4()
		{}

		HK_FORCE_INLINE explicit hkVectorT4(hkVector4Parameter v)
		{
			for (int i=0; i<4; ++i)
				m_quad[i] = (T) v(i);
		}

		HK_FORCE_INLINE void store(hkVector4& v)
		{
			for (int i=0; i<4; ++i)
				v(i) = (hkReal)m_quad[i];
		}

		HK_FORCE_INLINE void setSub(const hkVectorT4<T>& v0, const hkVectorT4<T>& v1)
		{
			for (int i=0; i<4; ++i)
				m_quad[i] = v0.m_quad[i] - v1.m_quad[i];
		}

		HK_FORCE_INLINE void setAddMul(const hkVectorT4<T>& a, const hkVectorT4<T>& m0, const hkVectorT4<T>& m1)
		{
			for (int i=0; i<4; ++i)
				m_quad[i] = a.m_quad[i] + m0.m_quad[i] * m1.m_quad[i];
		}

		HK_FORCE_INLINE void setAddMul(const hkVectorT4<T>& a, const hkVectorT4<T>& m0, const T& r)
		{
			for (int i=0; i<4; ++i)
				m_quad[i] = a.m_quad[i] + m0.m_quad[i] * r;
		}


		HK_FORCE_INLINE void setReciprocal(const hkVectorT4<T>& a)
		{
			for (int i=0; i<4; ++i)
				m_quad[i] = ((a.m_quad[i] == T(0)) ? hkTypeMax<T>() : T(1) / a.m_quad[i]);
		}

		HK_FORCE_INLINE void mul(const hkVectorT4<T>& a)
		{
			for (int i=0; i<4; ++i)
				m_quad[i] = m_quad[i] * a.m_quad[i];
		}

		HK_FORCE_INLINE void mul(const T& r)
		{
			for (int i=0; i<4; ++i)
				m_quad[i] = m_quad[i] * r;
		}

		template <int N>
		HK_FORCE_INLINE T dot(const hkVectorT4<T>& a) const
		{
			T sum(0);
			for (int i=0; i<N; ++i) 
				sum += (m_quad[i] * a.m_quad[i]);
			return sum;
		}

		template <int N>
		HK_FORCE_INLINE T lengthSquared() const
		{
			return dot<N>(*this);
		}

		template <int N>
		HK_FORCE_INLINE T getComponent() const
		{
			return m_quad[N];
		}

		T m_quad[4];
	};
	typedef hkVectorT4<hkDouble64> hkVectorD4;


	// Straight implementation using double precision
	HK_FORCE_INLINE hkInt32 accuracyReferenceRayCastSphere(const hkVectorD4 rayStart, 
														const hkVectorD4 dir, const hkVectorD4 spherePosAndRadius, 
														hkDouble64& hitFractionInOut, hkVectorD4& normalOut)
	{
		// Vector cr from sphere center to ray start
		hkVectorD4 cr; cr.setSub(rayStart, spherePosAndRadius);

		// Parameter b of the second degree intersection equation (projection of the un-normalized ray direction
		// on the un-normalized vector from center to ray start)
		hkDouble64 b = dir.dot<3>(cr);

		// Exit if ray pointing away from sphere	
		if (b >= 0)
		{
			return hkcdRayCastResult::createMiss();
		}

		// Parameter c of the second degree intersection equation (squared distance from center to ray start minus squared radius)
		hkDouble64 r = spherePosAndRadius.getComponent<3>();	
		HK_ASSERT2(0x02347FA3, r > 0, "sphere radius should not be zero");
		hkDouble64 c = cr.dot<3>(cr) - r * r;

		// Calculate simplified discriminant d (b^2 - a * c) and check if there is intersection
		hkDouble64 a = dir.lengthSquared<3>();
		hkDouble64 d = b * b - a * c;
		if (d < 0)
		{
			return hkcdRayCastResult::createMiss();
		}

		// Get earliest point of intersection t and check if it is between the ray start and the input hit fraction
		hkDouble64 t = -b - hkMath::sqrt(d);	
		if ((t < 0) || (t >= (a * hitFractionInOut)))	
		{
			return hkcdRayCastResult::createMiss();
		}	

		// Normalize hit fraction and calculate normal at intersection point	
		t = t / a;
		hkVectorD4 hitPoint; hitPoint.setAddMul(rayStart, dir, t);
		hkVectorD4 n; n.setSub(hitPoint, spherePosAndRadius);
		n.mul(hkDouble64(1) / hkMath::sqrt(n.lengthSquared<3>()));

		// Save output results
		hitFractionInOut = t;
		normalOut = n;
		return hkcdRayCastResult::createOutsideHit();
	}


	// Wrapper around double precision implementation taking hkReal parameters.
	HK_FORCE_INLINE	hkInt32 accuracyReferenceRayCastSphere(const hkcdRay& ray, hkVector4Parameter spherePosAndRadius, 
															hkSimdReal* HK_RESTRICT hitFractionInOut, hkVector4* HK_RESTRICT normalOut, hkcdRayQueryFlags::Enum flags)
	{
		hkDouble64 h = (hkDouble64) hitFractionInOut->getReal();
		hkVectorD4 n(*normalOut);

		hkInt32 ret = accuracyReferenceRayCastSphere(hkVectorD4(ray.m_origin), hkVectorD4(ray.getDirection()), hkVectorD4(spherePosAndRadius), h, n);

		hitFractionInOut->setFromFloat(hkReal(h));
		n.store(*normalOut);

		return ret;
	}
}


// Anonymous namespace for accuracy testing structs.
namespace {

	// Setup for an accuracy test case, from the setup, an actual test case can be constructed.
	// From the setup, test cases that have random values within the setup's ranges can be created.
	struct AccuracyTestCaseSetup
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DEMO, AccuracyTestCaseSetup);

		hkReal positionRange[2];	// Coordinate range for the positions of the sphere.
		hkReal radii[2];			// Radius range for the sphere radius.
		hkReal perturbation[2];		// Length of the perturbation for creating random rays around the tangent ray.
		hkReal distance[2];			// Distance range between the ray start position and the sphere position.
		hkReal dirScale;			// Scale of the unnormalized ray direction.
	
		// Stock setups provided for convenience.
		// They can be used to test specific situations for accuracy.
		// The semantics of the enum are:
		//	<Sphere center distance from origin>_<Ray start to sphere center distance>_<Ray direction length>
		enum EStockSetup
		{
			CLOSE_CLOSE_NORMALIZED, CLOSE_CLOSE, CLOSE_100_NORMALIZED, CLOSE_100, 
			CLOSE_1000_NORMALIZED, CLOSE_1000, CLOSE_10000_NORMALIZED, CLOSE_10000, 
			MEDIUM_CLOSE_NORMALIZED, MEDIUM_CLOSE, MEDIUM_100_NORMALIZED, MEDIUM_100,
			MEDIUM_1000_NORMALIZED, MEDIUM_1000, MEDIUM_10000_NORMALIZED, MEDIUM_10000, 
			FAR_CLOSE_NORMALIZED, FAR_CLOSE, FAR_100_NORMALIZED, FAR_100,
			FAR_1000_NORMALIZED, FAR_1000, FAR_10000_NORMALIZED, FAR_10000,
			STOCK_SETUP_COUNT
		};

		static const AccuracyTestCaseSetup sStockSetups[STOCK_SETUP_COUNT];

		// Initialize this setup from one of the stock setups.
		void initFromStock(EStockSetup setup)
		{
			*this = sStockSetups[setup];
		}
	};

	// Initialization of the stock setups.
	const AccuracyTestCaseSetup AccuracyTestCaseSetup::sStockSetups[] = 
	{ 
		{	{0.0f,0.01f}, {0.1f,0.1f}, {0.01f,0.01f}, {0.001f,1.0f}, 1.0f	},	// CLOSE_CLOSE_NORMALIZED
		{	{0.0f,0.1f}, {0.1f,0.1f}, {0.01f,0.01f}, {0.001f,1.0f}, 100.0f	},	// CLOSE_CLOSE
		{	{0.0f,0.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1.0f,100.0f}, 1.0f		},	// CLOSE_100_NORMALIZED
		{	{0.0f,0.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1.0f,100.0f}, 100.0f	},	// CLOSE_100
		{	{0.0f,0.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {100.0f,1000.0f}, 1.0f	},	// CLOSE_1000_NORMALIZED
		{	{0.0f,0.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {100.0f,1000.0f}, 100.0f	},	// CLOSE_1000
		{	{0.0f,0.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1000.0f,20000.0f}, 1.0f	},	// CLOSE_10000_NORMALIZED
		{	{0.0f,0.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1000.0f,20000.0f}, 100.0f	},	// CLOSE_10000

		{	{1000.0f,1000.01f}, {0.1f,0.1f}, {0.01f,0.01f}, {0.001f,1.0f}, 1.0f	},	// MEDIUM_CLOSE_NORMALIZED
		{	{1000.0f,1000.1f}, {0.1f,0.1f}, {0.01f,0.01f}, {0.001f,1.0f}, 100.0f	},	// MEDIUM_CLOSE
		{	{1000.0f,1000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1.0f,100.0f}, 1.0f	},	// MEDIUM_100_NORMALIZED
		{	{1000.0f,1000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1.0f,100.0f}, 100.0f	},	// MEDIUM_100
		{	{1000.0f,1000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {100.0f,1000.0f}, 1.0f	},	// MEDIUM_1000_NORMALIZED
		{	{1000.0f,1000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {100.0f,1000.0f}, 100.0f	},	// MEDIUM_1000
		{	{1000.0f,1000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1000.0f,20000.0f}, 1.0f	},	// MEDIUM_10000_NORMALIZED
		{	{1000.0f,1000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1000.0f,20000.0f}, 100.0f	},	// MEDIUM_10000

		{	{20000.0f,20000.01f}, {0.1f,0.1f}, {0.01f,0.01f}, {0.001f,1.0f}, 1.0f	},	// FAR_CLOSE_NORMALIZED
		{	{20000.0f,20000.1f}, {0.1f,0.1f}, {0.01f,0.01f}, {0.001f,1.0f}, 100.0f	},	// FAR_CLOSE
		{	{20000.0f,20000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1.0f,100.0f}, 1.0f		},	// FAR_100_NORMALIZED
		{	{20000.0f,20000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1.0f,100.0f}, 100.0f	},	// FAR_100
		{	{20000.0f,20000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {100.0f,1000.0f}, 1.0f	},	// FAR_1000_NORMALIZED
		{	{20000.0f,20000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {100.0f,1000.0f}, 100.0f	},	// FAR_1000
		{	{20000.0f,20000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1000.0f,20000.0f}, 1.0f	},	// FAR_10000_NORMALIZED
		{	{20000.0f,20000.5f}, {0.1f,0.1f}, {0.01f,0.01f}, {1000.0f,20000.0f}, 100.0f	},	// FAR_10000
	};


	// Generator for sampled setups in multiple dimensions.
	struct SampledAccuracySetupGen
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DEMO, SampledAccuracySetupGen);
		
		enum { DIM_COUNT = 4 };
		enum { MAX_MARKER_COUNT = 6 };

		// Advance the iterator to the next sample.
		static void advanceIterator(int dim, int interpolationCount, int iterator[DIM_COUNT], hkReal markers[DIM_COUNT][MAX_MARKER_COUNT], 
							int markerIndex[DIM_COUNT], int interpIndex[DIM_COUNT], int markerCount[DIM_COUNT])
		{
			++iterator[dim];

			markerIndex[dim] = iterator[dim] / interpolationCount;
			interpIndex[dim] = iterator[dim] - (markerIndex[dim] * interpolationCount);

			if (markerIndex[dim] + 1 == markerCount[dim])
			{
				iterator[dim] = 0;

				if (dim > 0)
					advanceIterator(dim-1, interpolationCount, iterator, markers, 
									markerIndex, interpIndex, markerCount);
				else
					iterator[dim] = -1;	// Done, Invalidate the iterator.
			}
		}

		static bool nextSetup(int iterator[DIM_COUNT], AccuracyTestCaseSetup& setup, int interpolationCount)
		{
			// The dimension types
			enum EDimType { DIM_SPACE, DIM_RADIUS, DIM_RANGE, DIM_DIRSCALE, DIM_COUNT };
			
			// The sampling markers, interpolationCount samples are taken between each 2 markers.
			hkReal markers[DIM_COUNT][MAX_MARKER_COUNT] = 
			{
				{ 0.1f,   1.0f, 10.0f, 100.0f, 1000.0f, 20000.0f },		// DIM_SPACE
				{ 0.1f,   1.0f, 10.0f, 100.0f,  100.0f,   100.0f },		// DIM_RADIUS
				{ 0.12f,  1.0f, 10.0f, 100.0f, 1000.0f, 20000.0f },	// DIM_RANGE
				{ 0.001f, 1.0f, 10.0f, 100.0f, 1000.0f, 20000.0f },	// DIM_DIRSCALE
			}; 

			int markerIndex[DIM_COUNT];
			int interpIndex[DIM_COUNT];
			int markerCount[DIM_COUNT] = { 6, 3, 6, 6 };

			// Calculate the current markers and interpolation indices
			// and exit if the iterator reached the end or is invalid.
			for (int d = 0 ; d < DIM_COUNT; ++d)
			{
				markerIndex[d] = iterator[d] / interpolationCount;
				interpIndex[d] = iterator[d] - (markerIndex[d] * interpolationCount);

				if (markerIndex[d] < 0 || markerIndex[d] + 1 >= markerCount[d])
					return false;

				if (interpIndex[d] < 0 || interpIndex[d] >= interpolationCount)
					return false;
			}

			// Fill the setup values.
			hkReal interpFactor;

			interpFactor = (hkReal) interpIndex[DIM_DIRSCALE] / (hkReal) interpolationCount;
			hkReal dirScaleSize = markers[DIM_DIRSCALE][markerIndex[DIM_DIRSCALE]+1] - markers[DIM_DIRSCALE][markerIndex[DIM_DIRSCALE]];
			setup.dirScale = markers[DIM_DIRSCALE][markerIndex[DIM_DIRSCALE]] + dirScaleSize * interpFactor;

			interpFactor = (hkReal) interpIndex[DIM_RANGE] / (hkReal) interpolationCount;
			hkReal rangeSize = markers[DIM_RANGE][markerIndex[DIM_RANGE]+1] - markers[DIM_RANGE][markerIndex[DIM_RANGE]];
			setup.distance[0] = markers[DIM_RANGE][markerIndex[DIM_RANGE]] + rangeSize * interpFactor;
			setup.distance[1] = setup.distance[0]; 

			interpFactor = (hkReal) interpIndex[DIM_RADIUS] / (hkReal) interpolationCount;
			hkReal radiusSize = markers[DIM_RADIUS][markerIndex[DIM_RADIUS]+1] - markers[DIM_RADIUS][markerIndex[DIM_RADIUS]];
			setup.radii[0] =  markers[DIM_RADIUS][markerIndex[DIM_RADIUS]] + radiusSize * interpFactor;
			setup.radii[1] = setup.radii[0]; 

			interpFactor = (hkReal) interpIndex[DIM_SPACE] / (hkReal) interpolationCount;
			hkReal spaceSize = markers[DIM_SPACE][markerIndex[DIM_SPACE]+1] - markers[DIM_SPACE][markerIndex[DIM_SPACE]];
			setup.positionRange[0] =  markers[DIM_SPACE][markerIndex[DIM_SPACE]] + spaceSize * interpFactor;
			setup.positionRange[1] = setup.positionRange[0]; 

			// Skip over setups where the ray start is inside the sphere, we are not interested in them.
			if (setup.distance[0] <= setup.radii[0])
			{
				++iterator[DIM_RANGE];
				return nextSetup(iterator, setup, interpolationCount);
			}

			// Advance the iterator
			advanceIterator(DIM_COUNT-1, interpolationCount, iterator, markers, 
							markerIndex, interpIndex, markerCount);

			return true;
		}
	};

	

	// An accuracy test case.
	// A test case is initialized from a setup, deciding on the random values used, and can 
	// then provide with random test rays.
	struct AccuracyTestCase
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DEMO, AccuracyTestCase);

		hkReal testRange;
		hkVector4 rayStart;
		hkVector4 sphere;
		hkVector4 tangentPoint;

		hkVector4 testRayPoint;
		hkVector4 testRayVector;
		hkcdRay testRay;

		// Initialize this test case based on the given setup.
		bool init(const AccuracyTestCaseSetup& setup, hkPseudoRandomGenerator& rGen)
		{
			testRange = rGen.getRandRange(setup.distance[0], setup.distance[1]);

			hkVector4 rayPosDir;
			rayPosDir.setZero();

			while (rayPosDir.equalZero().allAreSet())
			{
				rayPosDir(0) = rGen.getRandReal11();
				rayPosDir(1) = rGen.getRandReal11();
				rayPosDir(2) = rGen.getRandReal11();
			}

			rayPosDir.normalize<3>();

			// Initialize the sphere position
			sphere(0) = rGen.getRandRange(setup.positionRange[0],setup.positionRange[1]);
			sphere(1) = rGen.getRandRange(setup.positionRange[0],setup.positionRange[1]);
			sphere(2) = rGen.getRandRange(setup.positionRange[0],setup.positionRange[1]);
			sphere(3) = 0;

			// Initialize the ray start position
			rayStart.setAddMul(sphere, rayPosDir, hkSimdReal::fromFloat(testRange));

			// Make sure the test case is sane.
			hkVector4 objectsAparts;
			objectsAparts.setSub(rayStart, sphere);
			objectsAparts.zeroComponent<3>();
			if (objectsAparts.equalZero().allAreSet())
				return false;

			// Set the sphere radius
			sphere(3) = rGen.getRandRange(setup.radii[0], setup.radii[1]);
			
			// Initialize the ray end position as a random tangent point to the sphere.
			hkVector4 rayStartToSphereCenter;
			rayStartToSphereCenter.setSub(sphere, rayStart);
			rayStartToSphereCenter.zeroComponent<3>();

			// Calculate the angle by which we need to rotate ray to sphere center vectors 
			// so that they become tangent vectors.
			hkReal rotAngle = hkMath::asin(sphere(3) / rayStartToSphereCenter.length<3>().getReal());
			hkVector4 rotAxis;

			// Calculate a random rotation axis.
			{
				hkVector4 randDir;	rGen.getRandomVector11(randDir);
				randDir.setNormalizedEnsureUnitLength<3>( randDir );

				rotAxis.setCross(rayStartToSphereCenter, randDir);
				rotAxis.setNormalizedEnsureUnitLength<3>( rotAxis );
			}


			// Calculate the random tangent point using the exact rotation angle and the
			// the random axis.
			hkVector4 tangVec;
			hkRotation tangAxisRot; 
			tangAxisRot.setAxisAngle(rotAxis, rotAngle);
			tangAxisRot.multiplyVector(rayStartToSphereCenter, tangVec);
			tangentPoint.setAddMul(rayStart, tangVec, hkSimdReal::fromFloat(1.1f));

			return true;
		}

		
		// Sets up a new random test ray.
		void setupTestRay(const AccuracyTestCaseSetup& setup, hkPseudoRandomGenerator& rGen)
		{
			hkVector4 randDir;
			randDir.setZero();

			// Calculate a random direction.
			do 
			{
				rGen.getRandomVector11(randDir);

			} while(randDir.length<3>().isEqualZero());

			randDir.normalize<3>();
			randDir(3) = 0;

			// Calculate a perturbed point using the random direction.
			testRayPoint.setAddMul(tangentPoint, randDir, hkSimdReal::fromFloat(rGen.getRandRange(setup.perturbation[0], setup.perturbation[1])));
			testRayVector.setSub(testRayPoint, rayStart);

			// Set the ray parameters.
			hkVector4 rayDir = testRayVector;

			rayDir.normalize<3>();
			rayDir.mul(hkSimdReal::fromFloat(setup.dirScale));

			testRay.setOriginDirection(rayStart, rayDir);
		}
	};


	// Accuracy tester.
	// It runs a complete sampling based test between the declared functions and saves the results.
	struct AccuracyTester
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DEMO, AccuracyTester);

		// Enums for the test functions.
		enum ETestFunction
		{
			FUNC_REFERENCE,
			FUNC_CURRENT,
			FUNC_COUNT
		};

		// Prototypes for the test functions.
		typedef hkInt32 (*FuncImpl) (const hkcdRay&, hkVector4Parameter, hkSimdReal* HK_RESTRICT, hkVector4* HK_RESTRICT, hkcdRayQueryFlags::Enum flags);

		// Pointers to the implementations of the functions.
		static const FuncImpl sFuncs[FUNC_COUNT];
		// Descriptive names of the functions.
		static const char* const sFuncNames[FUNC_COUNT];

		struct Results
		{
			// Since we use random perturbations around tangent point, the hit to miss 
			// ratio of the reference function should be around 0.5, if not, something might be wrong with the tests.
			int refHits;							// The number of hits from the reference function 
			int refMisses;							// The number of misses from the reference function 

			int wrongHits[FUNC_COUNT];				// The number of wrong hits (compared to reference).
			int wrongMisses[FUNC_COUNT];			// The number of wrong misses (compared to reference).
			int wrongTolerantHits[FUNC_COUNT];		// The number of wrong hits after tolerance adjustments.
			int wrongTolerantMisses[FUNC_COUNT];	// The number of wrong misses after tolerance adjustments.	 
			float hitFractionErrAvg[FUNC_COUNT];	// The average error in hitFraction (compared to reference).
			int hitFractionErrAvgCount[FUNC_COUNT];	// The number of hitFraction error samples for calculating the running average.

			Results()
			{
				reset();
			}

			// Reset the results.
			void reset()
			{
				hkString::memSet(this, 0, sizeof(Results));
			}


			// Returns a string representation of the results.
			hkStringBuf toString()
			{
				hkStringBuf msg;
				msg.printf("%s h/m:%ld-%ld", sFuncNames[FUNC_REFERENCE], refHits, refMisses);

				for (int a = 1; a < FUNC_COUNT; ++a)
				{
					hkStringBuf funcMsg;
					funcMsg.printf(" <%s:%ld-%ld toler.:%ld-%ld fe:%f> ", sFuncNames[a], wrongHits[a], wrongMisses[a], wrongTolerantHits[a], wrongTolerantMisses[a], hitFractionErrAvg[a]);

					msg.append(funcMsg.cString());
				}

				return msg;
			}
		};

		// Run one test iteration step by generating 'caseCount' cases from the given setup, and for each testing 'rayCount' perturbed rays.
		static void runTestStep(const AccuracyTestCaseSetup& setup, hkPseudoRandomGenerator& rGen, Results& results, int caseCount, int rayCount, hkReal tolerance)
		{
			AccuracyTestCase test;

			for (int i = 0; i < caseCount; ++i)
			{
				test.init(setup, rGen);

				for (int j = 0; j < rayCount; ++j)
				{
					test.setupTestRay(setup, rGen);

					// Declare outputs.
					hkSimdReal hitFraction[FUNC_COUNT];
					hkVector4 hitNormal[FUNC_COUNT]; 
					hkBool32 didHit[FUNC_COUNT];

					// Initialize outputs.
					for (int a = 0; a < FUNC_COUNT; ++a)
					{
						hitFraction[a] = hkSimdReal::getConstant<HK_QUADREAL_MAX>();
						hitNormal[a].setZero();
					}

					// Call the test functions
					for (int a = 0; a < FUNC_COUNT; ++a)
					{
						didHit[a] = (*sFuncs[a])(test.testRay, test.sphere, &hitFraction[a], &hitNormal[a], hkcdRayQueryFlags::NO_FLAGS);
					}

					// Calculate the hitFraction error running averages.
					if (didHit[FUNC_REFERENCE])
					{
						for (int a = 1; a < FUNC_COUNT; ++a)
						{
							if (didHit[a])
							{
								hkReal err = hkMath::abs(hitFraction[FUNC_REFERENCE].getReal() - hitFraction[a].getReal());
								results.hitFractionErrAvg[a] = results.hitFractionErrAvg[a] + (((float)err - results.hitFractionErrAvg[a]) / (float) (results.hitFractionErrAvgCount[a] + 1));
								++results.hitFractionErrAvgCount[a];
							}
						}
					}

					// Update 'refHits' and 'refMisses'.
					if (didHit[FUNC_REFERENCE])
					{
						++results.refHits;
					}
					else
					{
						++results.refMisses;
					}

					// Create the tolerance sphere. When a function produces a wrong result, we run it again with 
					// a sphere that has a radius adjust by the given 'tolerane' and see if the problem disappears.
					hkVector4 largerSphere = test.sphere;
					hkVector4 smallerSphere = test.sphere;

					largerSphere(3) = largerSphere(3) + tolerance * largerSphere(3);
					smallerSphere(3) = smallerSphere(3) - tolerance * smallerSphere(3);

					// Update errors and re-call functions using the tolerance spheres.
					for (int a = 1; a < FUNC_COUNT; ++a)
					{
						if (bool(didHit[FUNC_REFERENCE]) != bool(didHit[a]))
						{
							if (didHit[a])
							{
								++results.wrongHits[a];

								hkBool32 didHitTolerant = (*sFuncs[a])(test.testRay, smallerSphere, &hitFraction[a], &hitNormal[a], hkcdRayQueryFlags::NO_FLAGS);

								if (didHitTolerant)
									++results.wrongTolerantHits[a];

							}
							else
							{
								++results.wrongMisses[a];

								hkBool32 didHitTolerant = (*sFuncs[a])(test.testRay, largerSphere, &hitFraction[a], &hitNormal[a], hkcdRayQueryFlags::NO_FLAGS);

								if (!didHitTolerant)
								{
									++results.wrongTolerantMisses[a];
								}
								else
								{

									{
										hkReal err = hkMath::abs(hitFraction[FUNC_REFERENCE].getReal() - hitFraction[a].getReal());
										results.hitFractionErrAvg[a] = results.hitFractionErrAvg[a] + (((float)err - results.hitFractionErrAvg[a]) / (float) (results.hitFractionErrAvgCount[a] + 1));
										++results.hitFractionErrAvgCount[a];
									}
								}

							}
						}
					}
				}
			}
		}


		// Run the full sampling test.
		// The ranges of the sampled dimensions and the distribution of samples are hard-coded in SampledAccuracySetupGen::nextSetup.
		static void runSamplingTest(hkPseudoRandomGenerator& rGen, Results& results, int interpolationCount, int caseCount, int rayCount, hkReal tolerance)
		{
			AccuracyTestCaseSetup setup;
			setup.initFromStock(AccuracyTestCaseSetup::CLOSE_CLOSE);

			int iterator[SampledAccuracySetupGen::DIM_COUNT];

			for (int d = 0; d < SampledAccuracySetupGen::DIM_COUNT; ++d){ 	iterator[d] = 0; }

			while (SampledAccuracySetupGen::nextSetup(iterator, setup, interpolationCount))
			{
				runTestStep(setup, rGen, results, caseCount, rayCount, tolerance);
			}
		}
	};

	const AccuracyTester::FuncImpl AccuracyTester::sFuncs[AccuracyTester::FUNC_COUNT] = { 
		accuracyReferenceRayCastSphere, 
		hkcdRayCastSphere, 
	};

	const char* const AccuracyTester::sFuncNames[AccuracyTester::FUNC_COUNT] = { 
		"Reference (double)", 
		"Current", 
	};

}


static void executeTest(const TestCase& testCase);
static void executeTest4x(const TestCase& testCase);
static void executeTestAccuracy();


// The test cases array can't be global because the memory manager has to be initialized before
// the variable can be defined
static void raySphereIntersectionTests()
{	
	// BEWARE, if the target sphere is not the same for all test cases the bundle version will fail!
	TestCase::myVector sphere(0, 0, 0, 1);		
	TestCase testCases[] = 
	{
		TestCase("ray outside", TestCase::myVector(-2, 2, 0), TestCase::myVector(2, 2, 0), sphere, hkcdRayQueryFlags::NO_FLAGS), 	
		TestCase("ray outside, penetrating twice", TestCase::myVector(-2, 0, 0), TestCase::myVector(2, 0, 0), sphere, hkcdRayQueryFlags::NO_FLAGS, hkSimdReal::getConstant<HK_QUADREAL_INV_4>(), TestCase::myVector(-1, 0, 0)),  

		TestCase("ray penetrating, start inside", TestCase::myVector(0, 0, 0), TestCase::myVector(2, 0, 0), sphere, hkcdRayQueryFlags::NO_FLAGS),
		TestCase("ray penetrating, start outside", TestCase::myVector(-2, 0, 0), TestCase::myVector(0, 0, 0), sphere, hkcdRayQueryFlags::NO_FLAGS, hkSimdReal::getConstant<HK_QUADREAL_INV_2>(), TestCase::myVector(-1, 0, 0)), 				 
		TestCase("ray completely inside", TestCase::myVector(-0.5f, 0, 0), TestCase::myVector(0.5f, 0, 0), sphere, hkcdRayQueryFlags::NO_FLAGS), 								
	
		// repeat all tests with inside hits on

		TestCase("ray outside", TestCase::myVector(-2, 2, 0), TestCase::myVector(2, 2, 0), sphere, hkcdRayQueryFlags::ENABLE_INSIDE_HITS), 	
		TestCase("ray outside, penetrating twice", TestCase::myVector(-2, 0, 0), TestCase::myVector(2, 0, 0), sphere, hkcdRayQueryFlags::ENABLE_INSIDE_HITS, hkSimdReal::getConstant<HK_QUADREAL_INV_4>(), TestCase::myVector(-1, 0, 0)),  

		TestCase("ray penetrating, start inside", TestCase::myVector(0, 0, 0), TestCase::myVector(2, 0, 0), sphere, hkcdRayQueryFlags::ENABLE_INSIDE_HITS, hkSimdReal::getConstant<HK_QUADREAL_INV_2>(), TestCase::myVector(1, 0, 0)),
		TestCase("ray penetrating, start outside", TestCase::myVector(-2, 0, 0), TestCase::myVector(0, 0, 0), sphere, hkcdRayQueryFlags::ENABLE_INSIDE_HITS, hkSimdReal::getConstant<HK_QUADREAL_INV_2>(), TestCase::myVector(-1, 0, 0)), 				 
		TestCase("ray completely inside", TestCase::myVector(-0.5f, 0, 0), TestCase::myVector(0.5f, 0, 0), sphere, hkcdRayQueryFlags::ENABLE_INSIDE_HITS), 								
	};	

	int numTestCases = sizeof(testCases) / sizeof(TestCase);

	// Execute all test cases with the single ray function
	for (int test = 0; test < numTestCases; ++test)
	{
		executeTest(testCases[test]);
		executeTest4x(testCases[test]);
	}	

	// Execute accuracy test
	executeTestAccuracy();
}

static void executeTestAccuracy()
{
	hkPseudoRandomGenerator rGen( 1234 );
	//hkPseudoRandomGenerator rGen((int) (hkSystemClock::getTickCounter()));
	
	AccuracyTester::Results results;
	AccuracyTester::runSamplingTest(rGen, results, 4, 2, 6, 0.01f);
	//HK_REPORT(results.toString().cString());

	HK_TEST2(results.wrongTolerantHits[AccuracyTester::FUNC_CURRENT] == 0, "hkcdRayCastSphere deteriorated");
	HK_TEST2(results.wrongTolerantMisses[AccuracyTester::FUNC_CURRENT] == 0, "hkcdRayCastSphere deteriorated");
	
	float totalErrors = (float) (results.wrongHits[AccuracyTester::FUNC_CURRENT] + results.wrongMisses[AccuracyTester::FUNC_CURRENT]);
	float totalRays = (float) (results.refHits + results.refMisses);

	HK_TEST2(totalErrors / totalRays < 0.002f, "hkcdRayCastSphere deteriorated");
}

static void executeTest(const TestCase& testCase)
{	
	// Execute the actual test		
	hkSimdReal tolerance; tolerance.setFromFloat(0.000001f);
	hkSimdReal hitFraction = hkSimdReal::getConstant<HK_QUADREAL_1>();
	hkVector4 normal; normal.setZero();

	hkcdRay ray;
	ray.setEndPoints(testCase.m_rayStart.data, testCase.m_rayEnd.data);
	hkBool32 result = hkcdRayCastSphere(ray, testCase.m_spherePosAndRadius.data, &hitFraction, &normal, testCase.m_queryFlags);	

	// Check results
	//HK_REPORT(testCase.m_name << ": [" << result << ", " << hitFraction.getReal() << ", " << normal << "]");
	HK_TEST2( bool(result) == bool(testCase.m_result), testCase.m_name);
	if (testCase.m_result == true) 
	{
		HK_TEST2((hitFraction - testCase.m_hitFraction).isLessEqual(tolerance), testCase.m_name);
		normal.sub(testCase.m_normal.data);
		HK_TEST2(normal.length<3>().isLessEqual(tolerance), testCase.m_name);
	}	
}

static void executeTest4x(const TestCase& testCase)
{	
	// flags are not supported for the bundle version
	if (testCase.m_queryFlags != hkcdRayQueryFlags::NO_FLAGS)
		return;

	// Execute the actual test		
	hkSimdReal tolerance; tolerance.setFromFloat(0.000001f);
	hkVector4 hitFraction = hkVector4::getConstant<HK_QUADREAL_1>();
	hkFourTransposedPoints normal;	

	hkcdRayBundle rayBundle;
	rayBundle.m_start.setAll(testCase.m_rayStart.data);
	rayBundle.m_end.setAll(testCase.m_rayEnd.data);
	rayBundle.m_activeRays.set<hkVector4ComparisonMask::MASK_XYZW>();

	hkVector4Comparison result = hkcdRayBundleSphereIntersect(rayBundle, testCase.m_spherePosAndRadius.data.getComponent<3>(), hitFraction, normal);	

	// Check results
	//HK_REPORT(testCase.name << ": [" << result << ", " << hitFraction.getReal() << ", " << normal << "]");
	HK_TEST2( (result.allAreSet()!=hkFalse32) == testCase.m_result, testCase.m_name << ": " << testCase.m_result);
	if (testCase.m_result == true) 
	{
		HK_TEST2(hkMath::equal(hitFraction(0), testCase.m_hitFraction.getReal()), testCase.m_name);
		//HK_TEST2((hitFraction - testCase.hitFraction).lessEqual(tolerance).allAreSet(), testCase.name);
		//normal.sub(testCase.normal.data);
		//HK_TEST2(normal.length<3>().lessEqual(tolerance).allAreSet(), testCase.name);
	}	
}


int RaySphereTests_main()
{	
	raySphereIntersectionTests();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(RaySphereTests_main, "Fast", "Geometry/Test/UnitTest/Internal/", __FILE__ );

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
