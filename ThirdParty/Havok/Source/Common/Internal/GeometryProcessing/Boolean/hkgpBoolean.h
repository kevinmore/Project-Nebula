/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKGP_BOOLEAN_H
#define HKGP_BOOLEAN_H

#include <Common/Base/Config/hkProductFeatures.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>

///
/// Boolean operations on polyhedra.
/// The following inputs properties must hold:
///
///		For classification dependant operators (from A_MINUS_B to A_OR_B):
///		- Non self-intersecting.
///		- Closed 2-manifold.
///		Note: theses requirements only apply the result of an operator, not to the operands themselves.
///
/// Inputs triangle materials is preserved during operations to allow for properties tracking / remapping.
///
/// Notes: You can retrieve the owner of a output triangle by checking the first two indices as follow:
///			if index0 < index1 then this triangle belong to the operand B else to the operand A.
///
struct hkgpBoolean
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpBoolean);
	/// Operators.
	struct Operator {
		enum _ {
		EMPTY,									///< Evaluate to empty set.
		A_MINUS_B,								///< Evaluate A - B.
		B_MINUS_A,								///< Evaluate B - A.
		A_AND_B,								///< Evaluate A & B.
		A_OR_B,									///< Evaluate A | B.
		A01,									///< Refine A.
		B01,									///< Refine B.
		A0,										///< A out.
		A1,										///< A in.
		B0,										///< B out.
		B1,										///< B in.
		NUM_OPERATORS
	}; static const char* HK_CALL getName(_); };

	/// Operand.
	struct Operand
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpBoolean::Operand);
		enum eType
		{
			TYPE_GEOMETRY,	///< Operand is a geometry.
			TYPE_PLANE,		///< Operand is a plane.
		};
		inline	Operand()											{ clear(); }
		inline	Operand(const hkGeometry* g)						{ clear(); m_geometry=g; }
		inline	Operand(const hkTransform& t,const hkGeometry* g)	{ clear(); m_transform.set(t);m_geometry=g; }
		inline	Operand(const hkMatrix4& t,const hkGeometry* g)		{ clear(); m_transform=t;m_geometry=g; }
		inline	Operand(const hkVector4& worldPlane)				{ clear(); m_plane=worldPlane; m_type=TYPE_PLANE; }
		void	clear()												{ m_type=TYPE_GEOMETRY; m_geometry=HK_NULL; m_transform.setIdentity(); m_flipOrientation = false; }
		hkBool	operator==(const Operand& other) const;

		eType				m_type;				///< Operand type
		hkMatrix4			m_transform;		///< Operand world transformation.
		const hkGeometry*	m_geometry;			///< Operand geometry.
		hkVector4			m_plane;			///< Operand plane.
		hkBool				m_flipOrientation;	///< Flip operand inside out.
	};
	
	/// Error type.
	struct ErrorType { 
		enum _ {
		INVALID_TRIANGLE,						///< Invalid triangle found in the input.
		DUPLICATED_VERTEX,						///< Two intersection point at the same position.
		COPLANAR_VERTEX,						///< Vertex lies exactly in a plane of the other operand.
		BAD_VERTEX,								///< Cannot triangulate face due to bad vertex.
		BAD_EDGE,								///< Cannot triangulate face due to bad edge (always come in pairs).
		BAD_TRIANGLE,							///< Cannot triangulate face boundaries due to bad triangle.
		INVALID_PARTITION,						///< Cannot partition face.
		INVALID_LOOP_ENDPOINT,					///< Open intersection loop end-point. Note: this is not a critical error if operands are open.
		NUM_ERRORS
	}; static const char* HK_CALL getName(_); };


	/// Error descriptor.
	struct Error
	{
		Error(ErrorType::_ type, int subType, const hkVector4& pos) : m_type(type),m_subType(subType),m_order(0),m_position(pos) {}
		
		ErrorType::_	m_type;					///< Type of the error.
		int				m_subType;				///< Sub type of the error (internal).
		int				m_order;				///< Greater than zero if multiple errors at the same position.
		hkVector4		m_position;				///< Position of the error (valid only if W is 1).
	};	

	/// Evaluation configuration.
	struct Config
	{
		Config()	:	m_alwaysBuildOutput(false)
					,	m_checkLoops(true)
					,	m_classThreshold(0)
					,	m_maxError(0)
					,	m_maxIterations(0)
					,	m_maxErrorsCount(8)
					,	m_errorTarget(1)
					,	m_useExactArithmetic(false)
					,	m_processComponents(false) {}

		hkBool			m_alwaysBuildOutput;	///< Always build an output even if error(s) occurs.
		hkBool			m_checkLoops;			///< Perform intersection loops check.
		hkReal			m_classThreshold;		///< Classification threshold, set to zero to use fast predicate.
		hkReal			m_maxError;				///< Maximum allowed error.
		int				m_maxIterations;		///< Maximum conflicts solver iterations.
		int				m_maxErrorsCount;		///< Maximum number of errors to record.
		int				m_errorTarget;			///< Operand to perturb if required.
		bool			m_useExactArithmetic;	///< Use exact computation whenever possible, very slow.
		bool			m_processComponents;	///< Process each operand components separately.
	};

	/// Evaluation informations.
	struct Infos
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpBoolean::Infos);
		Infos() : m_intersect(false),m_iterations(0),m_hasCriticalErrors(false),m_errorLength(0) {}

		hkArray<Error>	m_errors;				///< Errors generated during the operation.
		hkBool			m_intersect;			///< True if the two operands intersect.
		int				m_iterations;			///< Number of iterations used.
		hkBool			m_hasCriticalErrors;	///< Critical errors occurs.
		hkReal			m_errorLength;			///< Magnitude of the error generated during the operation.
	};

	/// Evaluator.
	struct Evaluator
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,Evaluator);
		virtual			~Evaluator() {}
		virtual hkBool	hasErrors() const=0;
	};

	//
	// API
	//
	
	/// Prepare for delayed evaluation.
	static Evaluator* HK_CALL	prepare(const Operator::_* ops, int numOps, const Operand& a, const Operand& b, const Config& config=Config(), Infos* infosOut=HK_NULL);

	/// Evaluate an operator.
	static hkBool HK_CALL		evaluate(const Evaluator* evaluator, Operator::_ op, hkGeometry& result);
	
	/// Direct evaluation of multiple operators at once.
	static hkBool HK_CALL		evaluate(const Operator::_* ops, int numOps, const Operand& a, const Operand& b, hkArray<hkGeometry*>& results, const Config& config=Config(), Infos* infosOut=HK_NULL);
	
	/// Direct evaluation of a single operator.
	/// \note\a result is allowed to be a reference to \a a or \a b geometry.
	static hkBool HK_CALL		evaluate(Operator::_ op, const Operand& a, const Operand& b, hkGeometry& result, const Config& config=Config(), Infos* infosOut=HK_NULL);
};

#endif // HKGP_BOOLEAN_H

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
