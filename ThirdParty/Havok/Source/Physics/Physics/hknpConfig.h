/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONFIG_H
#define HKNP_CONFIG_H

//
// Compile time configuration constants.
//

#define HKNP_SHAPE_DEFAULT_CONVEX_RADIUS 0.01f
#define HKNP_SHAPE_MIN_SCALE 0.001f
#define HKNP_SHAPE_BUFFER_SIZE 2048
#define HKNP_MAX_SIZEOF_SHAPE 384
#define HKNP_DEFAULT_TRIANGLE_DEGENERACY_TOLERANCE 1e-7f

// HKNP_MAX_NUM_MANIFOLDS_PER_BATCH must be a power of 2 and has a minimum of 2 (because of silhouette manifolds).
// HKNP_NUM_MXJACOBIANS_PER_BLOCK multiplied by sizeof(hknpMxContactJacobian) must exactly equal the block steam block data size.
// HKNP_CONTACT_JACOBIAN_IS_COMPRESSED is currently only supported in combination with HKNP_NUM_MX_JACOBIANS = 4.
// HKNP_ENABLE_SOLVER_PARALLEL_TASKS is currently only supported for UMA architectures.

#if defined(HK_PLATFORM_HAS_SPU) && defined(HK_TINY_SPU_ELF)

	#define HKNP_ENABLE_SOLVER_PARALLEL_TASKS 0
	#define HKNP_CONTACT_JACOBIAN_IS_COMPRESSED 0
	#define HKNP_NUM_MX_JACOBIANS 1
	#define HKNP_INTEGRATOR_MX 4

	enum
	{
		HKNP_MAX_NUM_MATERIALS_ON_SPU = 8,
		HKNP_MAX_NUM_MOTION_PROPERTIES_ON_SPU = 8,
		HKNP_MAX_NUM_MANIFOLDS_PER_BATCH = 4,
		HKNP_NUM_MXJACOBIANS_PER_BLOCK = 1,
	};


#elif defined(HK_PLATFORM_HAS_SPU)

	#define HKNP_ENABLE_SOLVER_PARALLEL_TASKS 0
	#define HKNP_CONTACT_JACOBIAN_IS_COMPRESSED 0
	#define HKNP_NUM_MX_JACOBIANS 1
	#define HKNP_INTEGRATOR_MX 4

	// Solver export : solver results writer buffer sizes
	#define HKNP_SPU_SOLVE_RESULTS_WRITER_BASE_BUFFER_SIZE 512
	#define HKNP_SPU_SOLVE_RESULTS_WRITER_OVERFLOW_BUFFER_SIZE 128

	#define HKNP_SPU_SOLVE_RESULTS_WRITER_DMA_GROUP 0

	// Disabling SDF collisions gives us ~14KB of extra space in the collide elf
	#define HKNP_DISABLE_SIGNED_DISTANCE_FIELD_COLLISIONS_ON_SPU

	// Disabling height fields gives us ~6KB of extra space in the collide elf
	#define HKNP_DISABLE_HEIGHT_FIELDS_ON_SPU

	// Disabling the welding modifier gives us ~4KB of extra space in the collide elf
	//#define HKNP_DISABLE_WELDING_MODIFIER_ON_SPU

	// Save around ~30Kb of code size in query elf
	#define HKNP_DISABLE_UNSCALLED_QUERY_ON_SPU

	enum
	{
		HKNP_MAX_NUM_MATERIALS_ON_SPU = 128,
		HKNP_MAX_NUM_MOTION_PROPERTIES_ON_SPU = 128,
		HKNP_MAX_NUM_MANIFOLDS_PER_BATCH = 2,
		HKNP_NUM_MXJACOBIANS_PER_BLOCK = 2,
		HKNP_MAX_SHAPE_KEY_ITERATOR_SIZE_ON_SPU = 64,
		HKNP_COMPOUND_HIERARCHY_BUFFER = 1024,			// The additional amount of memory allocated for storing recursive hknpCompoundShapes (without the actual leaf shape)
		HKNP_MAX_SHAPE_KEY_MASK_SIZE_ON_SPU = 64,
	};


#elif defined(HK_PLATFORM_XBOX360)

	//#define HKNP_CVX_COLLIDE_INLINE

	#define HKNP_ENABLE_SOLVER_PARALLEL_TASKS 1
	#define HKNP_CONTACT_JACOBIAN_IS_COMPRESSED 0
	#define HKNP_NUM_MX_JACOBIANS 1
	#define HKNP_INTEGRATOR_MX 4

	enum
	{
		HKNP_MAX_NUM_MANIFOLDS_PER_BATCH = 128,
	};

#else

	#define HKNP_ENABLE_SOLVER_PARALLEL_TASKS 1
	#define HKNP_CONTACT_JACOBIAN_IS_COMPRESSED 0
	#define HKNP_NUM_MX_JACOBIANS 1
	#define HKNP_INTEGRATOR_MX 4
	#define HKNP_ENABLE_CONTACT_SOLVER_DYNAMIC_STATIC_OPTIMIZATION

	enum
	{
		HKNP_MAX_NUM_MANIFOLDS_PER_BATCH = 128,
	};

#endif


// SPU buffer constants
enum
{
#if defined(HK_TINY_SPU_ELF)
	HKNP_MAX_SHAPE_SIZE_ON_SPU = 1024,	// around 64 verts
	HKNP_SPU_UNTYPED_NUM_CACHE_ROWS = 1,
#else
	HKNP_MAX_SHAPE_SIZE_ON_SPU = 2048,	// around 64 verts
	HKNP_SPU_UNTYPED_NUM_CACHE_ROWS = 4,
#endif
	HKNP_SPU_UNTYPED_CACHE_LINE_SIZE = 256,
};

//#define HKNP_ENABLE_SOLVER_LOG
#if defined(HKNP_ENABLE_SOLVER_LOG)
#	define HKNP_ON_SOLVER_LOG(X) X
#else
#	define HKNP_ON_SOLVER_LOG(X)
#endif

// Use the hknpJobQueue based solver by default on all platforms except PlayStation(R)3.
// The lockless solver may be more efficient when there are no other processes competing for the cores.
#if defined(HK_PLATFORM_HAS_SPU)
	#define HKNP_ENABLE_LOCKLESS_SOLVER
#endif

#endif // HKNP_CONFIG_H

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
