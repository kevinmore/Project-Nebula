/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_VIRTUAL_MACHINE_H

// Virtual Machine command definitions

//#define HK_MOPP_DEBUGGER_ENABLED
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>

#ifdef HK_MOPP_DEBUGGER_ENABLED
#	include <Physics2012/Internal/Collide/Mopp/Builder/hkbuilder.h>
#	include <Physics2012/Internal/Collide/Mopp/Utility/hkpMoppDebugger.h>
#	define HK_QVM_DBG(x) x
#	define HK_QVM_DBG2(var,x) int var = x
#else 
#	define HK_QVM_DBG(x)
#	define HK_QVM_DBG2(var,x) 
#endif


//this is the structure that all virtual machines will return on being queried
//it represents a primitive ID and an array of primitive properties
class hkpMoppPrimitiveInfo
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppPrimitiveInfo );

		hkUint32 ID;
		//unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];
};

//class HK_ALIGNED_VARIABLE(hkpMoppVirtualMachine,16) 
class hkpMoppVirtualMachine
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppVirtualMachine );

		typedef int hkpMoppFixedPoint;  

	public:
		// standard constructor
		HK_FORCE_INLINE hkpMoppVirtualMachine();										
		// standard destructor
		HK_FORCE_INLINE ~hkpMoppVirtualMachine();										

		HK_FORCE_INLINE int getNumHits() const ;

	protected:

		HK_FORCE_INLINE void addHit(unsigned int id, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES]);

#if !defined(HK_PLATFORM_SPU)

		HK_FORCE_INLINE void initQuery( hkArray<hkpMoppPrimitiveInfo>* m_primitives_out );

#else

		inline void initQuery( hkpMoppPrimitiveInfo* m_primitives_out, int primitives_out_capacity );

#endif
			/// returns an integer which is smaller than x
		static HK_FORCE_INLINE int HK_CALL toIntMin(hkReal x);

			/// returns an integer which is larger than x
		static HK_FORCE_INLINE int HK_CALL toIntMax(hkReal x);

		static HK_FORCE_INLINE int HK_CALL read24( const unsigned char* PC );


	public:

#if !defined(HK_PLATFORM_SPU)
		hkArray<hkpMoppPrimitiveInfo>* m_primitives_out;
		int		padding[3];
#else
		hkPadSpu<hkpMoppPrimitiveInfo*> m_primitives_out;
		hkPadSpu<int>				   m_primitives_idx;
		hkPadSpu<int>					m_primitives_out_capacity;
#endif

};

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.inl>

#endif // HK_COLLIDE2_MOPP_VIRTUAL_MACHINE_H

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
