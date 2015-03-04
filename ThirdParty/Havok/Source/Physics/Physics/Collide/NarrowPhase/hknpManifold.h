/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MANIFOLD_H
#define HKNP_MANIFOLD_H

#include <Geometry/Internal/Types/hkcdManifold4.h>

struct hknpManifoldCollisionCache;
class hknpMaterial;

struct hknpManifoldBase : public hkcdManifold4
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpManifoldBase );

	/// The type of manifold.
	enum ManifoldType
	{
		TYPE_NORMAL,	///< this is a normal manifold
		TYPE_TRIGGER,	///< this is a trigger. Skips friction calculations.
		TYPE_TOI,		///< this turns the contact into a ghost plane and the body movement is rolled back so it does not penetrate this plane
	};

	template <bool pad, typename T> struct PadType				{};
	template <typename T>			struct PadType<true, T>		{	typedef hkPadSpu<T> Type;	};
	template <typename T>			struct PadType<false, T>	{	typedef T Type;				};
};

#define HKNP_PADDED_TYPE(T)	typename Pad<T>::Type

/// Temporary collision detection data. Used to build a Jacobian.
template <bool pad>
struct hknpPaddedManifold : public hknpManifoldBase
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpPaddedManifold );
		template <typename T> struct Pad { typedef typename PadType<pad, T>::Type Type; };

	public:

		/// Initialization helper
		void init( hkVector4Parameter normal, const hkVector4* pointAndDistances, int numPoints );

		/// Initialization helper
		void init( hkVector4Parameter normal, hkVector4Parameter position, hkReal distance );

		/// Check the consistency
		void checkConsistency() const;

		/// Copies another manifold over this one
		template <bool otherPad>
		HK_FORCE_INLINE void copy(const hknpPaddedManifold<otherPad>& other);

	public:

		/// Number of valid points in m_positions, between 1 and 4.
		/// Note that even if numPoints < 4, all values in the hkcdManifold4 have to be initialized.
		HKNP_PADDED_TYPE(int) m_numPoints;

		/// Manifold type. See ManifoldType.
		hkEnum<ManifoldType,hkUint8> m_manifoldType;

		/// If set to one if the solver should run extra iteration to reduce penetration.
		HKNP_PADDED_TYPE(hkUint8) m_useIncreasedIterations;

		/// Set to 1 if a the manifold is either new or has significantly changed.
		/// This is used to control how the friction error from the last frame is used for this frame.
		HKNP_PADDED_TYPE(hkUint8) m_isNewSurface;

		/// Set to 1 if this manifold is a new manifold and set to 0 if this manifold is older.
		HKNP_PADDED_TYPE(hkUint8) m_isNewManifold;

		/// object A invMass will be multiplied with 1+m_massChangerData, object b invMass with 1-m_massChangerData.
		/// So if
		///   - m_massChangerData == 0.0f   no change
		///   - m_massChangerData > 0.0f and  m_massChangerData < 1.0f   object A will become lighter
		///   - m_massChangerData < 0.0f and  m_massChangerData > -1.0f  object A will become heavier
		HKNP_PADDED_TYPE(hkReal) m_massChangerData;

		/// The material of objectB (material of objectA is implicit)
		HKNP_PADDED_TYPE(const hknpMaterial*) m_materialB;

		/// The shape keys of the shapes involved.
		HKNP_PADDED_TYPE(hknpShapeKey) m_shapeKeyA;
		HKNP_PADDED_TYPE(hknpShapeKey) m_shapeKeyB;

		/// Optional collision cache.
		HKNP_PADDED_TYPE(hknpManifoldCollisionCache*) m_collisionCache;

		/// Optional collision cache in main memory, where applicable (e.g. on SPU).
		HKNP_PADDED_TYPE(hknpManifoldCollisionCache*) m_collisionCacheInMainMemory;
};

typedef hknpPaddedManifold<true>	hknpSpuManifold;
typedef hknpPaddedManifold<false>	hknpPpuManifold;

#ifdef HK_PLATFORM_SPU
struct hknpManifold : public hknpSpuManifold {};
#else
struct hknpManifold : public hknpPpuManifold {};
#endif

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.inl>

#endif // HKNP_MANIFOLD_H

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
