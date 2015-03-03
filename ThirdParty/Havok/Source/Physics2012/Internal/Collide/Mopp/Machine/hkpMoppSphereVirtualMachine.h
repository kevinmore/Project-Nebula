/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_SPHERE_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_SPHERE_VIRTUAL_MACHINE_H


#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>

class hkpMoppSphereVirtualMachine : public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppSphereVirtualMachine );

		struct hkpMoppSphereVirtualMachineQuery
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppSphereVirtualMachine::hkpMoppSphereVirtualMachineQuery );

			// this order must be preserved: x,y,z,radius
			int m_x;		
			int m_y;
			int m_z;
			int m_radius;

			//the offset of the all previous scales are accumulated here
			int m_offset_x;		
			int m_offset_y;
			int m_offset_z;
			// the current offset for the primitives
			unsigned int m_primitiveOffset;  
			
			//the shifts from all previous scale commands are accumulated here
			int m_shift;		

			unsigned int m_properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];
		};

	public:

		// standard constructor
		inline hkpMoppSphereVirtualMachine();										
		// standard destructor
		inline ~hkpMoppSphereVirtualMachine();										

		/// run a query, the results are added to the hitlist
		// returns the number of hits
		void querySphere(const hkpMoppCode* code, const hkSphere &sphere, hkArray<hkpMoppPrimitiveInfo>* primitives_out);

		/// run a query on a P4 system
		// returns the number of hits
		void querySphereP4(const hkpMoppCode* code, const hkSphere &sphere, hkArray<hkpMoppPrimitiveInfo>* primitives_out);


	protected:
		//
		//these are for the sphere query
		//
		// this order must be preserved: x,y,z,radius
		HK_ALIGN16( hkpMoppFixedPoint m_x );		
		hkpMoppFixedPoint m_y;
		hkpMoppFixedPoint m_z;
		hkpMoppFixedPoint m_radius;

		void querySphereOnTree	( const hkpMoppSphereVirtualMachineQuery* query, const unsigned char* commands);
};

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppSphereVirtualMachine.inl>

#endif // HK_COLLIDE2_MOPP_SPHERE_VIRTUAL_MACHINE_H

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
