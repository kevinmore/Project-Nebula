/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_AABB_CAST_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_AABB_CAST_VIRTUAL_MACHINE_H

// Virtual Machine command definitions
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>
#include <Physics2012/Collide/Shape/hkpShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

// Virtual Machine command definitions
struct hkpMoppAabbCastVirtualMachineQueryInt
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppAabbCastVirtualMachineQueryInt );

	//the offset of the all previous scales are accumulated here (in B space)
	hkVector4 m_FtoBoffset;
	hkVector4 m_extents;	// in B space
	hkReal	  m_extentsSum3;

	//the shifts from all previous scale commands are accumulated here
	int m_shift;

	// this converts  floating point space into form byte space
	hkReal m_FtoBScale;

	// the current offset for the primitives
	unsigned int m_primitiveOffset;  
	unsigned int m_properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];
};

struct hkpMoppAabbCastVirtualMachineQueryFloat
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppAabbCastVirtualMachineQueryFloat );

	// this is the ray in local int coordinate space
	hkVector4 m_rayEnds[2];
};

/// This class implements an AABB cast within the MOPP.
/// This is pretty much identical to the hkpMoppAabbCastVirtualMachine,
/// so make sure that both versions are kept in sync.
class hkpMoppAabbCastVirtualMachine: public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppAabbCastVirtualMachine );

		inline hkpMoppAabbCastVirtualMachine(){}
		inline ~hkpMoppAabbCastVirtualMachine(){}

		/// Input structure to the AABB cast.
		struct hkpAabbCastInput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppAabbCastVirtualMachine::hkpAabbCastInput );

			/// The starting position in MOPP space.
			hkVector4 m_from;

			/// the destination position in MOPP space.
			hkVector4 m_to;

			/// The halfExtents of the AABB in MOPP space.
			hkVector4 m_extents;


			/// collision input, only used in addRealHit to forward calls.
			HK_PAD_ON_SPU( const hkpLinearCastCollisionInput* ) m_collisionInput;

			/// the body which is encapsulated in the AABB, only used in addRealHit to forward calls.
			HK_PAD_ON_SPU( const hkpCdBody* ) m_castBody;

			/// the body containing the MOPP space only used in addRealHit to forward calls.
			HK_PAD_ON_SPU( const hkpCdBody* ) m_moppBody;
		};

		/// This functions casts an AABB through the MOPP space and for every hit
		/// it gets the child shape from the shapeCollection and call linearCast.
		void aabbCast( const hkpAabbCastInput& input, hkpCdPointCollector& castCollector, hkpCdPointCollector* startCollector );

	////////////////////////////////////////////////////////////////
	//
	// THE REMAINDER OF THIS FILE IS FOR INTERNAL USE
	//
	//////////////////////////////////////////////////////////////// 
#if defined(HK_PLATFORM_SPU)
		HK_PAD_ON_SPU( int )                 m_dmaGroup;
		HK_PAD_ON_SPU( const hkUint8* )      m_originalBaseAddress;
#endif

	protected:

		/// the MOPP code, remember it has the very important info struct inside.
		HK_PAD_ON_SPU( const hkpMoppCode* )      m_code;

		/// converts from 24 bit int space into floating point space.
		HK_PAD_ON_SPU( float )                  m_ItoFScale;

		hkpShapeType                             m_castObjectType;
		
		/// set to the current hitFraction.
		HK_PAD_ON_SPU( hkReal )                 m_earlyOutFraction;
		HK_PAD_ON_SPU( hkReal )                 m_refEarlyOutFraction;

		/// for being able to forward calls.
		HK_PAD_ON_SPU( hkpCdPointCollector* )    m_castCollector;
		HK_PAD_ON_SPU( hkpCdPointCollector* )    m_startPointCollector;

		HK_PAD_ON_SPU( const hkpAabbCastInput* ) m_input;

		// If set (-1) this mask ensures that the chunkId is stored in the shape key
		HK_PAD_ON_SPU( int )                    m_reindexingMask;

		void queryRayOnTree	( const hkpMoppAabbCastVirtualMachineQueryInt* query, const unsigned char* commands,hkpMoppAabbCastVirtualMachineQueryFloat* const fQuery, int chunkOffset);

		/// This function forwards a hit to a child cast.
		/// You can look into its implementation to see what's going on.
		/// (Maybe this function should be virtual.)
#if ! (defined(HK_ARCH_ARM) || defined(HK_PLATFORM_SPU))
		HK_FORCE_INLINE 
#endif 
		void addHit(unsigned int id, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES], hkUlong chunkId=0);

};

#endif // HK_COLLIDE2_MOPP_AABB_CAST_VIRTUAL_MACHINE_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
