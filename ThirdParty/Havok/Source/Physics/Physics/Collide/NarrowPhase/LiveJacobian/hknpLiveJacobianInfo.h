/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_LIVE_JACOBIAN_INFO_H
#define HKNP_LIVE_JACOBIAN_INFO_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>

struct hknpConvexConvexManifoldCollisionCache;


/// Helper struct to trigger collision detection during solving
struct hknpLiveJacobianInfo
{
	enum Type
	{
		SINGLE_CVX_CVX,
		CHILD_CVX_CVX,
		CHILD_CVX_CVX_LAST_IN_BATCH,
	};

	void initLiveJacobian( const hknpManifold* manifold, const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB, Type type = SINGLE_CVX_CVX );

	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpLiveJacobianInfo );
	HK_ALIGN16(const hknpConvexConvexManifoldCollisionCache* m_cache);
	HK_PAD_ON_SPU( hknpMxContactJacobian* ) m_jacobian[2];
	HK_PAD_ON_SPU( hkUint8 ) m_substepOfLastBuildJac;
	HK_PAD_ON_SPU( hkUint8 ) m_indexOfManifoldInJacobian[2];
	HK_PAD_ON_SPU( hkUint8 ) m_numManifolds;
	hkEnum<Type, hkUint8> m_type;

	/// If the body travel distance is smaller than this, no recollide will be triggered.
	hkHalf m_currentDistance;		// +overrideType(hkUint16)

	/// The maximum distance the body can move without causing a recollide.
	hkHalf m_maxLinearMovement;		// +overrideType(hkUint16)

	/// The maximum angle in radians the body can move without causing a recollide.
	hkHalf m_maxAngularMovement;	// +overrideType(hkUint16)
};

class hknpLiveJacobianInfoStream: public hkBlockStream<hknpLiveJacobianInfo>{};
class hknpLiveJacobianInfoWriter: public hkBlockStream<hknpLiveJacobianInfo>::Writer{};
class hknpLiveJacobianInfoReader: public hkBlockStream<hknpLiveJacobianInfo>::Reader{};
class hknpLiveJacobianInfoModifier: public hkBlockStream<hknpLiveJacobianInfo>::Modifier{};


/// A live Jacobian range can point to other ranges using LinkedRange.m_next forming a linked list.
/// In this case, all the ranges correspond to the same entry in the containing grid.
class hknpLiveJacobianInfoRange : public hkBlockStreamBase::LinkedRange
{
	public:

		hknpLiveJacobianInfoRange(): hkBlockStreamBase::LinkedRange()
		{
			m_processedIter = 0;
		}

	public:

		int m_processedIter;
};

class hknpLiveJacobianInfoGrid: public hknpGrid<hknpLiveJacobianInfoRange> {};

#endif // HKNP_COLLISION_CACHE_H

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
