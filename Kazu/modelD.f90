!-----------------------------------------------------------------------
! module model D
! 
! This module is a library for calcuration quantities related with model D
! such as visibilities, bi-spectrum, amplitude, closure phases,....,etc
!
! Writer  : Kazunori Akiyama
! Version : 20120721
! 
! dependency:
!   module: basic_models (basic_models.f90)
!-----------------------------------------------------------------------
module modelD
    use basic_models
    implicit none
    private     :: pi
    real(8)     :: pi    = atan(1.0d0)*4.0d0 ! common variable in this module
contains
    ! 2 Component Circular Gaussian !
    !    p(1)  : S1
    !    p(2)  : phi1
    !    p(3)  : S2/S1
    !    p(4)  : r2
    !    p(5)  : theta2
    !    p(6)  : phi2
    subroutine calc_visibility (u,v,p,Vcomp,n)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n
        real(8),    intent(in)  :: u(1:n),v(1:n) ! uv coverage
        real(8),    intent(in)  :: p(1:6)       ! parameters
        complex(8), intent(out) :: Vcomp(1:n)    ! visibility
        
        ! Local variables !
        real(8)    :: c1p(1:4), c2p(1:4)!, c3p(1:4)  ! parameters for each gaussian
        complex(8) :: Vcomp1(1:n), Vcomp2(1:n)!, Vcomp3(1:n)     ! visibilities of each gaussian
        !real(8)    :: size0
        
        !size0 = pi/180.0d0/3600.0d0/1000.0d0
        
        ! parameters for each gaussian
        c1p(1:4)  = (/ p(1), 0.0d0, 0.0d0, p(2)  /)
        c2p(1:4)  = (/ p(3)*p(1), p(4),  p(5),  p(6)  /)
        
        call circular_gaussian(u,v,c1p,Vcomp1,n)
        call circular_gaussian(u,v,c2p,Vcomp2,n)
        
        Vcomp = Vcomp1 + Vcomp2
    end subroutine
    
    ! Pcheck
    subroutine check_p (pmin,pmax,p,flag)
        implicit none
        real(8),    intent(in)     :: pmax(1:6),pmin(1:6)
        real(8),    intent(inout)  :: p(1:6)
        integer,    intent(out)    :: flag
        
        integer :: i
        
        flag = 1
        
        ! Adjust Phase
        p(5) = atan2(sin(p(5)),cos(p(5)))
        
        do i=1,6
            if (p(i) < pmin(i) .or. p(i) > pmax(i)) then
                flag = -1
            end if
        end do
        
    end subroutine check_p
    
    ! Calcurate Bispectrum
    subroutine calc_bispectrum (u1,v1,u2,v2,u3,v3,p,BScomp,n)
        implicit none
        
        ! Arguments !
        integer,    intent(in)    :: n
        real(8),    intent(in)    :: u1(1:n),v1(1:n)           ! uv coverage
        real(8),    intent(in)    :: u2(1:n),v2(1:n)           ! uv coverage
        real(8),    intent(in)    :: u3(1:n),v3(1:n)           ! uv coverage
        real(8),    intent(in)    :: p(1:6)                   ! parameters
        complex(8),    intent(out)    :: BScomp(1:n)               ! visibility
        
        ! Local variables !
        complex(8)    :: Vcomp1(1:n), Vcomp2(1:n), Vcomp3(1:n) ! visibilities of each gaussian
        
        ! parameters for each gaussian
        call calc_visibility (u1,v1,p,Vcomp1,n)
        call calc_visibility (u2,v2,p,Vcomp2,n)
        call calc_visibility (u3,v3,p,Vcomp3,n)
        
        BScomp = Vcomp1 * Vcomp2 * Vcomp3
    end subroutine
    
    ! Calcurate Visibility Amplitude
    subroutine calc_amp (u,v,p,Vamp,n)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n
        real(8),    intent(in)  :: u(1:n),v(1:n) ! uv coverage
        real(8),    intent(in)  :: p(1:6)       ! parameters
        real(8),    intent(out) :: Vamp(1:n)     !
        
        ! Local variables !
        complex(8)    :: Vcomp(1:n)    ! visibility
        
        call calc_visibility (u,v,p,Vcomp,n)
        Vamp = abs(Vcomp)
        
    end subroutine
    
    ! Calcurate Closure Phase
    subroutine calc_cphase (u1,v1,u2,v2,u3,v3,p,BSphase,n)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n
        real(8),    intent(in)  :: u1(1:n),v1(1:n) ! uv coverage
        real(8),    intent(in)  :: u2(1:n),v2(1:n) ! uv coverage
        real(8),    intent(in)  :: u3(1:n),v3(1:n) ! uv coverage
        real(8),    intent(in)  :: p(1:6)         ! parameters
        real(8),    intent(out) :: BSphase(1:n)    ! Closure Phase
        
        ! Local variables !
        complex(8)    :: BScomp(1:n)                 ! bispectrum
        
        call calc_bispectrum (u1,v1,u2,v2,u3,v3,p,BScomp,n)
        BSphase = atan2(aimag(BScomp),real(BScomp))
        !BSphase = atan2(real(BScomp),aimag(BScomp))
    end subroutine
    
    ! Calcurate Reduced chi-square
    subroutine calc_chisquare  (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,dof,chisq,rchisq,n,m)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n,m
        real(8),    intent(in)  :: u (1:n), v(1:n)               ! uv coverage
        real(8),    intent(in)  :: Vamp(1:n),Vamp_var(1:n)       ! Amplitude & its variance
        
        real(8),    intent(in)  :: u1(1:m),v1(1:m)               ! uv coverage
        real(8),    intent(in)  :: u2(1:m),v2(1:m)               ! uv coverage
        real(8),    intent(in)  :: u3(1:m),v3(1:m)               ! uv coverage
        real(8),    intent(in)  :: BSphase(1:m),BSphase_var(1:m) ! Closure Phase & its variance
        
        real(8),    intent(in)  :: p(1:6)         ! parameters
        
        integer,    intent(out) :: dof          ! deg. of free
        real(8),    intent(out) :: chisq,rchisq ! Chi-square & Reduced Chi-Square
        
        ! local variable
        real(8)    :: BSphase_model(1:m)    ! Model of Closure Phase
        real(8)    :: Vamp_model(1:n)       ! Model of Amplitude
        real(8)    :: dBS(1:m)
        
        ! calcurate model values
        call calc_cphase (u1,v1,u2,v2,u3,v3,p,BSphase_model,m)
        call calc_amp    (u,v,p,Vamp_model,n)
        
        ! calcurate residual of closure phases
        dBS = BSphase_model-BSphase
        dBS = atan2(sin(dBS),cos(dBS))
        
        ! calcurate chi-square
        chisq   =  sum( (Vamp - Vamp_model)**2 / Vamp_var ) + sum( (dBS)**2 / BSphase_var )
        dof     =  n+m-6+1
        rchisq  =  chisq/dof
    end subroutine
    
    subroutine calc_chisq  (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,chisq,n,m)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n,m
        real(8),    intent(in)  :: u (1:n), v(1:n)               ! uv coverage
        real(8),    intent(in)  :: Vamp(1:n),Vamp_var(1:n)       ! Amplitude & its variance
        
        real(8),    intent(in)  :: u1(1:m),v1(1:m)               ! uv coverage
        real(8),    intent(in)  :: u2(1:m),v2(1:m)               ! uv coverage
        real(8),    intent(in)  :: u3(1:m),v3(1:m)               ! uv coverage
        real(8),    intent(in)  :: BSphase(1:m),BSphase_var(1:m) ! Closure Phase & its variance
        
        real(8),    intent(in)  :: p(1:6)         ! parameters
        
        real(8),    intent(out) :: chisq
        
        ! local variable
        real(8)    :: BSphase_model(1:m)    ! Model of Closure Phase
        real(8)    :: Vamp_model(1:n)       ! Model of Amplitude
        real(8)    :: dBS(1:m)
        
        ! calcurate model values
        call calc_cphase (u1,v1,u2,v2,u3,v3,p,BSphase_model,m)
        call calc_amp    (u,v,p,Vamp_model,n)
        
        ! calcurate residual of closure phases
        dBS = BSphase_model-BSphase
        dBS = atan2(sin(dBS),cos(dBS))
        
        ! calcurate chi-square
        chisq   =  sum( (Vamp - Vamp_model)**2 / Vamp_var ) + sum( (dBS)**2 / BSphase_var )
    end subroutine
    
    ! Calcurate log likelihood
    subroutine calc_log_likelihood  (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,L,n,m)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n,m
        real(8),    intent(in)  :: u (1:n), v(1:n)               ! uv coverage
        real(8),    intent(in)  :: Vamp(1:n),Vamp_var(1:n)       ! Amplitude & its variance
        
        real(8),    intent(in)  :: u1(1:m),v1(1:m)               ! uv coverage
        real(8),    intent(in)  :: u2(1:m),v2(1:m)               ! uv coverage
        real(8),    intent(in)  :: u3(1:m),v3(1:m)               ! uv coverage
        real(8),    intent(in)  :: BSphase(1:m),BSphase_var(1:m) ! Closure Phase & its variance
        
        real(8),    intent(in)  :: p(1:6)         ! parameters
        
        real(8),    intent(out) :: L ! Chi-square & Reduced Chi-Square
        
        ! local variable
        integer :: dof          ! deg. of free
        real(8) :: chisq,rchisq ! Chi-square & Reduced Chi-Square
        
        ! Calcurate Chi-square
        call calc_chisquare  (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,dof,chisq,rchisq,n,m)
        
        ! Calcurate likelihood
        L = -0.50d0 * ( (n+m)*log(2*pi) + sum(log(Vamp_var)) + sum(log(BSphase_var)) + chisq )
        
    end subroutine 
    
    
    ! Calcurate AIC
    subroutine calc_AIC (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,AIC,n,m)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n,m
        real(8),    intent(in)  :: u (1:n), v(1:n)               ! uv coverage
        real(8),    intent(in)  :: Vamp(1:n),Vamp_var(1:n)       ! Amplitude & its variance
        
        real(8),    intent(in)  :: u1(1:m),v1(1:m)               ! uv coverage
        real(8),    intent(in)  :: u2(1:m),v2(1:m)               ! uv coverage
        real(8),    intent(in)  :: u3(1:m),v3(1:m)               ! uv coverage
        real(8),    intent(in)  :: BSphase(1:m),BSphase_var(1:m) ! Closure Phase & its variance
        
        real(8),    intent(in)  :: p(1:6)         ! parameters
        
        real(8),    intent(out) :: AIC ! AIC
        
        ! local variable
        real(8) :: L                   ! Likelihood
        
        call calc_log_likelihood  (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,L,n,m)
        
        AIC = -2*L + 6 * 2
        
    end subroutine!
    
    ! Calcurate BIC
    subroutine calc_BIC (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,BIC,n,m)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n,m
        real(8),    intent(in)  :: u (1:n), v(1:n)               ! uv coverage
        real(8),    intent(in)  :: Vamp(1:n),Vamp_var(1:n)       ! Amplitude & its variance
        
        real(8),    intent(in)  :: u1(1:m),v1(1:m)               ! uv coverage
        real(8),    intent(in)  :: u2(1:m),v2(1:m)               ! uv coverage
        real(8),    intent(in)  :: u3(1:m),v3(1:m)               ! uv coverage
        real(8),    intent(in)  :: BSphase(1:m),BSphase_var(1:m) ! Closure Phase & its variance
        
        real(8),    intent(in)  :: p(1:6)         ! parameters
        
        real(8),    intent(out) :: BIC ! BIC
        
        ! local variable
        real(8) :: L                   ! Likelihood
        
        call calc_log_likelihood  (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,p,L,n,m)
        
        BIC = -2*L + 6 * log(real(n+m))
        
    end subroutine
    
    ! Calcurate three quantities
    subroutine calc_3q (u,v,Vamp,Vamp_var,u1,v1,u2,v2,u3,v3,BSphase,BSphase_var,dof,chisq,rchisq,p,L,AIC,BIC,n,m)
        implicit none
        
        ! Arguments !
        integer,    intent(in)  :: n,m
        real(8),    intent(in)  :: u (1:n), v(1:n)               ! uv coverage
        real(8),    intent(in)  :: Vamp(1:n),Vamp_var(1:n)       ! Amplitude & its variance
        
        real(8),    intent(in)  :: u1(1:m),v1(1:m)               ! uv coverage
        real(8),    intent(in)  :: u2(1:m),v2(1:m)               ! uv coverage
        real(8),    intent(in)  :: u3(1:m),v3(1:m)               ! uv coverage
        real(8),    intent(in)  :: BSphase(1:m),BSphase_var(1:m) ! Closure Phase & its variance
        
        real(8),    intent(in)  :: p(1:6)         ! parameters
        
        integer,    intent(out) :: dof          ! deg. of free
        real(8),    intent(out) :: chisq,rchisq ! Chi-square & Reduced Chi-Square
        real(8),    intent(out) :: L ! Chi-square & Reduced Chi-Square
        real(8),    intent(out) :: AIC ! AIC
        real(8),    intent(out) :: BIC ! BIC
        
        real(8) :: Vamp_model(1:n)
        real(8) :: BSphase_model(1:m),dBS(1:m)
        
        ! calcurate model values
        call calc_cphase (u1,v1,u2,v2,u3,v3,p,BSphase_model,m)
        call calc_amp    (u,v,p,Vamp_model,n)
        
        ! calcurate residual of closure phases
        dBS = BSphase_model-BSphase
        dBS = atan2(sin(dBS),cos(dBS))
        
        ! calcurate chi-square
        chisq   =  sum( (Vamp - Vamp_model)**2 / Vamp_var ) + sum( (dBS)**2 / BSphase_var )
        dof     =  n+m-6+1
        rchisq  =  chisq/dof
        
        ! Calcurate likelihood
        L = -0.50d0 * ( (n+m)*log(2*pi) + sum(log(Vamp_var)) + sum(log(BSphase_var)) + chisq )
        
        ! Calcurate AIC
        AIC = -2*L + 6 * 2
        
        ! Calcurate BIC
        BIC = -2*L + 6 * log(real(n+m))
        
    end subroutine
end module modelD