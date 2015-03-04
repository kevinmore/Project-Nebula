/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//
// Havok Memory Optimised Partial Polytope Debugger
// This class helps debugging the MOPP assembler and virtual machine
//

#ifndef HK_COLLIDE2_MOPP_FIND_ALL_H
#define HK_COLLIDE2_MOPP_FIND_ALL_H

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>


// the idea of the debugger is that we are searching our triangle in the entire original tree and
// remembering all paths to this node.

// when it comes to the final virtual machine, we easily can check, which paths are not taken,
// and which extra paths are processed which shouldn't be processed
class hkpMoppFindAllVirtualMachine : public hkpMoppVirtualMachine
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppFindAllVirtualMachine );

			// standard constructor
		inline hkpMoppFindAllVirtualMachine(){}
		// standard destructor
		inline ~hkpMoppFindAllVirtualMachine(){}

		HK_FORCE_INLINE void queryAll(const hkpMoppCode* code, hkArray<hkpMoppPrimitiveInfo>* primitives_out);

	public: 
		struct hkpMoppFindAllVirtualMachineQuery
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppFindAllVirtualMachine::hkpMoppFindAllVirtualMachineQuery );

			unsigned int m_primitiveOffset;  
			unsigned int m_properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];
		};

		void queryOnTree	( const hkpMoppFindAllVirtualMachineQuery* query, const unsigned char* commands);
		void queryOnTreeLeft	( const hkpMoppFindAllVirtualMachineQuery* query, const unsigned char* commands);
		void queryOnTreeRight	( const hkpMoppFindAllVirtualMachineQuery* query, const unsigned char* commands);
};

void hkpMoppFindAllVirtualMachine::queryAll(const hkpMoppCode* code, hkArray<hkpMoppPrimitiveInfo>* primitives_out)
{
	m_primitives_out = primitives_out;
	hkpMoppFindAllVirtualMachineQuery query;
	query.m_primitiveOffset = 0;
	query.m_properties[0] = 0;
	queryOnTree( &query, &code->m_data[0]);
}

#endif // HK_COLLIDE2_MOPP_DEBUGGER_H

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
